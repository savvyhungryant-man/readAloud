// =====================================================================
//  PaddleOCR (PP-OCRv4) 브라우저용 – 사진 속 글자 인식
//   1) 글자 줄 찾기(detection): 사진에서 "글자가 있는 영역" 확률 지도를 만듦
//   2) 줄마다 잘라내기: 영역 덩어리마다 기울기를 계산해 반듯하게 잘라냄
//   3) 글자 읽기(recognition): 잘라낸 줄 이미지를 글자로 바꿈
//  OpenCV 없이 순수 자바스크립트로 구현 (브라우저와 Node 모두에서 동작)
// =====================================================================

export const MODEL_BASE = 'https://cdn.jsdelivr.net/npm/@gutenye/ocr-models@1.4.2/assets/';

export async function createOcr(ort, { base = MODEL_BASE, onProgress = () => {} } = {}) {
  const load = async (name, label) => {
    onProgress(label);
    const buf = await (await fetch(base + name)).arrayBuffer();
    return ort.InferenceSession.create(new Uint8Array(buf));
  };
  const det = await load('ch_PP-OCRv4_det_infer.onnx', '글자 찾기 모델 받는 중… (5MB)');
  const rec = await load('ch_PP-OCRv4_rec_infer.onnx', '글자 읽기 모델 받는 중… (11MB)');
  onProgress('사전 받는 중…');
  const dictText = await (await fetch(base + 'ppocr_keys_v1.txt')).text();
  const keys = ['', ...dictText.split(/\r?\n/).filter((k, i, a) => !(i === a.length - 1 && k === '')), ' '];
  return { ort, det, rec, keys };
}

// img = { data: RGBA 바이트, width, height }  (사진, 긴 변 2000px 정도로 줄여서 넣음)
export async function recognize(ocr, img, { detMaxSide = 1280, minScore = 0.5 } = {}) {
  const t0 = now();
  const boxes = await detect(ocr, img, detMaxSide);
  const t1 = now();
  const lines = [];
  for (const b of boxes) {
    const r = await readLine(ocr, img, b);
    if (r.text.trim() && r.score >= minScore) lines.push({ ...b, ...r });
  }
  const t2 = now();
  lines.sort((a, b) => a.cy - b.cy);
  return { lines, detMs: t1 - t0, recMs: t2 - t1 };
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

// ---------- 1) 글자 줄 찾기 ----------
async function detect({ ort, det }, img, maxSide) {
  const s = Math.min(1, maxSide / Math.max(img.width, img.height));
  const W = Math.max(32, Math.round(img.width * s / 32) * 32);
  const H = Math.max(32, Math.round(img.height * s / 32) * 32);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  const input = new Float32Array(3 * W * H);
  const sx = img.width / W, sy = img.height / H;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const px = sample(img, (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5);
    for (let c = 0; c < 3; c++) input[c * W * H + y * W + x] = (px[c] / 255 - mean[c]) / std[c];
  }
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, H, W]) });
  const prob = out[det.outputNames[0]].data;   // H*W, 0~1

  // 확률 0.3 넘는 점들을 이어진 덩어리(줄)로 묶기
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) mask[i] = prob[i] > 0.3 ? 1 : 0;
  const label = new Int32Array(W * H);
  const stack = new Int32Array(W * H);
  const boxes = [];
  let id = 0;
  for (let start = 0; start < W * H; start++) {
    if (!mask[start] || label[start]) continue;
    id++;
    let sp = 0, n = 0, sumX = 0, sumY = 0, sxx = 0, syy = 0, sxy = 0, sumP = 0;
    stack[sp++] = start; label[start] = id;
    const pts = [];
    while (sp) {
      const p = stack[--sp], x = p % W, y = (p - x) / W;
      pts.push(x, y); n++; sumX += x; sumY += y; sumP += prob[p];
      if (x > 0     && mask[p - 1] && !label[p - 1]) { label[p - 1] = id; stack[sp++] = p - 1; }
      if (x < W - 1 && mask[p + 1] && !label[p + 1]) { label[p + 1] = id; stack[sp++] = p + 1; }
      if (y > 0     && mask[p - W] && !label[p - W]) { label[p - W] = id; stack[sp++] = p - W; }
      if (y < H - 1 && mask[p + W] && !label[p + W]) { label[p + W] = id; stack[sp++] = p + W; }
    }
    if (n < 40 || sumP / n < 0.5) continue;                  // 너무 작거나 확신 낮은 덩어리는 버림
    const mx = sumX / n, my = sumY / n;
    for (let k = 0; k < pts.length; k += 2) {
      const dx = pts[k] - mx, dy = pts[k + 1] - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    // 덩어리의 기울기(주축 방향)와 그 방향으로의 길이·두께
    let ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    let ux = Math.cos(ang), uy = Math.sin(ang);
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (let k = 0; k < pts.length; k += 2) {
      const dx = pts[k] + 0.5 - mx, dy = pts[k + 1] + 0.5 - my;
      const u = dx * ux + dy * uy, v = -dx * uy + dy * ux;
      if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v;
    }
    let w = u1 - u0 + 1, h = v1 - v0 + 1;
    if (h > w) {   // 세로로 긴 덩어리 → 90도 돌려 가로 기준으로
      [w, h] = [h, w]; ang += Math.PI / 2; [ux, uy] = [-uy, ux];
      [u0, u1, v0, v1] = [v0, v1, -u1, -u0];
    }
    if (h < 4) continue;
    // 중심(투영 범위의 가운데), 원래 사진 좌표로 환산
    const cu = (u0 + u1) / 2, cv = (v0 + v1) / 2;
    const cxDet = mx + 0.5 + cu * ux - cv * uy, cyDet = my + 0.5 + cu * uy + cv * ux;
    // 찾은 영역은 실제 글자보다 얇게 나오므로 넓혀 줌 (PaddleOCR의 unclip과 같은 역할)
    const hh = h * 1.9, ww = w + hh * 0.6;
    boxes.push({ cx: cxDet * sx, cy: cyDet * sy, w: ww * sx, h: hh * sy, ang,
                 left: (cxDet - ww / 2 * Math.abs(ux) - hh / 2 * Math.abs(uy)) * sx });
  }
  return boxes;
}

// ---------- 2)+3) 줄 잘라내서 읽기 ----------
async function readLine({ ort, rec, keys }, img, b) {
  const H = 48;
  const W = Math.max(16, Math.min(2400, Math.round(H * b.w / b.h)));
  const cos = Math.cos(b.ang), sin = Math.sin(b.ang);
  const input = new Float32Array(3 * H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const u = ((x + 0.5) / W - 0.5) * b.w, v = ((y + 0.5) / H - 0.5) * b.h;
    const px = sample(img, b.cx + u * cos - v * sin - 0.5, b.cy + u * sin + v * cos - 0.5);
    for (let c = 0; c < 3; c++) input[c * H * W + y * W + x] = (px[c] / 255 - 0.5) / 0.5;
  }
  const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, H, W]) });
  const t = out[rec.outputNames[0]];
  const [, T, C] = t.dims, d = t.data;
  const em = [];   // 읽힌 글자와 그 위치(칸 번호)
  let prev = 0, sum = 0;
  for (let i = 0; i < T; i++) {
    let best = 0, bv = -Infinity;
    for (let c = 0; c < C; c++) if (d[i * C + c] > bv) { bv = d[i * C + c]; best = c; }
    if (best !== 0 && best !== prev) { em.push({ i, ch: keys[best] ?? '' }); sum += bv; }
    prev = best;
  }
  return { text: fixGaps(em, T, gray(input, H, W), H, W), score: em.length ? sum / em.length : 0 };
}

// ---------- 빠진 대시(—)·띄어쓰기 되살리기 ----------
// 이 모델은 긴 대시(—)를 못 읽고 건너뛰고, 글자 간격이 넓은 책에서는 띄어쓰기도 가끔 빠뜨림.
// 1) 읽힌 글자 사이 간격이 그 줄의 보통 간격보다 크게 벌어진 곳만 후보로 봄
// 2) 그 빈 곳의 실제 픽셀을 확인: 가운데 높이에 얇은 가로줄만 있으면 "—", 비어 있으면 띄어쓰기
function gray(input, H, W) {
  const g = new Float32Array(H * W);
  for (let i = 0; i < H * W; i++) g[i] = (input[i] + input[H * W + i] + input[2 * H * W + i]) / 3;   // -1(검정) ~ 1(흰색)
  return g;
}
// 빈 곳(x0~x1)을 세로로 훑어서 어떤 모양인지 판단
function gapShape(g, H, W, x0, x1, ink) {
  x0 = Math.max(0, Math.round(x0)); x1 = Math.min(W - 1, Math.round(x1));
  let cols = 0, empty = 0, thinMid = 0;
  for (let x = x0; x <= x1; x++) {
    let n = 0, top = H, bot = -1;
    for (let y = 4; y < H - 4; y++) if (g[y * W + x] < ink) { n++; if (y < top) top = y; if (y > bot) bot = y; }
    cols++;
    if (n === 0) empty++;
    else if (n <= H * 0.12 && bot - top <= H * 0.14 && (top + bot) / 2 > H * 0.33 && (top + bot) / 2 < H * 0.7) thinMid++;
  }
  return { cols, empty: empty / Math.max(1, cols), thinMid: thinMid / Math.max(1, cols) };
}
// x0~x1 사이에서 글자 잉크가 하나도 없는 세로줄이 가장 길게 이어진 폭 (픽셀)
function emptyRun(g, H, W, x0, x1, ink) {
  x0 = Math.max(0, Math.round(x0)); x1 = Math.min(W - 1, Math.round(x1));
  let best = 0, run = 0;
  for (let x = x0; x <= x1; x++) {
    let has = false;
    for (let y = 4; y < H - 4 && !has; y++) if (g[y * W + x] < ink) has = true;
    run = has ? 0 : run + 1;
    if (run > best) best = run;
  }
  return best;
}
function fixGaps(em, T, g, H, W) {
  if (!em.length) return '';
  const steps = [];
  for (let k = 1; k < em.length; k++) if (em[k].ch !== ' ' && em[k - 1].ch !== ' ') steps.push(em[k].i - em[k - 1].i);
  steps.sort((a, b) => a - b);
  const typical = steps.length ? steps[Math.floor(steps.length / 2)] : 3;
  const sorted = Float32Array.from(g).sort();
  const ink = (sorted[Math.floor(sorted.length * 0.02)] + sorted[Math.floor(sorted.length * 0.6)]) / 2;
  const px = t => (t + 0.5) * W / T;
  // 진짜 띄어쓰기 폭 기준: 모델이 읽어낸 띄어쓰기 자리의 빈 폭 중앙값
  const spaceRuns = [];
  for (let k = 1; k + 1 < em.length; k++) if (em[k].ch === ' ' && em[k - 1].ch !== ' ' && em[k + 1].ch !== ' ')
    spaceRuns.push(emptyRun(g, H, W, px(em[k - 1].i), px(em[k + 1].i), ink));
  spaceRuns.sort((a, b) => a - b);
  const spaceW = spaceRuns.length ? spaceRuns[Math.floor(spaceRuns.length / 2)] : H * 0.2;
  let text = em[0].ch;
  for (let k = 1; k < em.length; k++) {
    const a = em[k - 1], b = em[k];
    const gap = b.i - a.i;
    if (a.ch !== ' ' && b.ch !== ' ' && gap >= 3) {
      const run = emptyRun(g, H, W, px(a.i), px(b.i), ink);
      const sh = gap >= typical * 1.7 && gap >= 4
        ? gapShape(g, H, W, px(a.i) + (px(b.i) - px(a.i)) * 0.3, px(a.i) + (px(b.i) - px(a.i)) * 0.7, ink) : null;
      if (typeof globalThis.__gapDebug === 'function') globalThis.__gapDebug({ left: text.slice(-8), right: b.ch, gap, typical, run, spaceW, thin: sh ? sh.thinMid : -1 });
      if (sh && sh.thinMid >= 0.6) text += '—';
      else if (run >= spaceW * 0.6 && /[A-Za-z0-9.,;:!?'’"”)]/.test(a.ch) && /[A-Za-z0-9"“(]/.test(b.ch)) text += ' ';
    }
    text += b.ch;
  }
  return text;
}

// 사진에서 (x, y) 위치의 색을 주변 4점 보간으로 가져오기 (바깥은 흰색)
function sample(img, x, y) {
  const { data, width: w, height: h } = img;
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const r = [0, 0, 0];
  for (let k = 0; k < 4; k++) {
    const xx = x0 + (k & 1), yy = y0 + (k >> 1);
    const wgt = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy);
    if (xx < 0 || yy < 0 || xx >= w || yy >= h) { r[0] += 255 * wgt; r[1] += 255 * wgt; r[2] += 255 * wgt; continue; }
    const i = (yy * w + xx) * 4;
    r[0] += data[i] * wgt; r[1] += data[i + 1] * wgt; r[2] += data[i + 2] * wgt;
  }
  return r;
}

// 인식된 줄들 → 글 (들여쓰기로 문단 나눔)
export function linesToText(lines) {
  if (!lines.length) return '';
  const lefts = lines.map(l => l.left).sort((a, b) => a - b);
  const baseLeft = lefts[Math.floor(lefts.length / 2)];
  const out = [];
  for (const l of lines) {
    if (out.length && l.left - baseLeft > l.h * 0.5) out.push('');
    out.push(l.text);
  }
  return out.join('\n');
}
