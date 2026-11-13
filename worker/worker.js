// Read Aloud – OpenAI TTS 프록시 (Cloudflare Worker)
// 비밀 값(Settings → Variables and Secrets):
//   OPENAI_API_KEY : OpenAI API 키 (sk-...)
//   APP_PASSWORD   : 앱에서 입력할 비밀번호

const ALLOWED_ORIGIN = 'https://savvyhungryant-man.github.io';  // 이 주소의 페이지만 호출 허용
const MODEL          = 'gpt-4o-mini-tts';
const MAX_CHARS      = 1000;                                    // 한 번에 읽을 최대 글자 수 (요금 폭탄 방지)
const VOICES = ['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse'];
const INSTRUCTIONS =
  'You are reading a storybook aloud to a young child who is learning English. ' +
  'Speak warmly, clearly and naturally, with gentle expression and natural pauses.';
// 말하는 빠르기 (앱의 속도 슬라이더에 따라 3단계)
const PACE = {
  normal:    '',
  slow:      ' Speak noticeably more slowly than normal conversation, about 85% of your usual speed. ' +
             'Keep this slower pace from the first word to the very last word. Do not speed up.',
  very_slow: ' Speak slowly and very clearly, about 70% of your usual speed, like reading to a beginner reader, ' +
             'with short pauses between phrases. Keep this slow pace from the first word to the very last word. Do not speed up.',
};
// 실험: OpenAI의 speed 값도 함께 보냄 (모델이 지원하지 않으면 무시되거나 오류 → 앱 디버그 로그에 표시됨)
const SPEED = { normal: 1.0, slow: 0.85, very_slow: 0.7 };

function cors(origin) {
  return {
    'Access-Control-Allow-Origin': origin === ALLOWED_ORIGIN ? origin : ALLOWED_ORIGIN,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-App-Password',
    'Access-Control-Max-Age': '86400',
  };
}

function json(obj, status, origin) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', ...cors(origin) },
  });
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';

    // 브라우저가 실제 요청 전에 보내는 사전 확인
    if (request.method === 'OPTIONS') return new Response(null, { headers: cors(origin) });

    // 상태 확인용 (브라우저에서 주소를 열면 보임). 비밀 값 내용은 절대 보여주지 않음
    if (request.method === 'GET') {
      return json({ ok: true, hasApiKey: !!env.OPENAI_API_KEY, hasPassword: !!env.APP_PASSWORD }, 200, origin);
    }

    if (request.method !== 'POST') return json({ error: 'method not allowed' }, 405, origin);

    // ① 허용된 페이지에서 온 요청인지
    if (origin !== ALLOWED_ORIGIN) return json({ error: 'origin not allowed' }, 403, origin);

    // ② 비밀번호가 맞는지
    if (!env.APP_PASSWORD || request.headers.get('X-App-Password') !== env.APP_PASSWORD) {
      return json({ error: 'wrong password' }, 401, origin);
    }

    // ③ 입력 검사
    let body;
    try { body = await request.json(); } catch { return json({ error: 'bad json' }, 400, origin); }
    const text  = String(body.text || '').trim();
    const voice = VOICES.includes(body.voice) ? body.voice : 'coral';
    const pace  = Object.hasOwn(PACE, body.pace) ? body.pace : 'normal';
    if (!text)                   return json({ error: 'empty text' }, 400, origin);
    if (text.length > MAX_CHARS) return json({ error: 'text too long' }, 413, origin);

    // ④ OpenAI 호출 (API 키는 여기서만 붙음)
    const r = await fetch('https://api.openai.com/v1/audio/speech', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${env.OPENAI_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: MODEL, input: text, voice, response_format: 'mp3',
        instructions: INSTRUCTIONS + PACE[pace],
        ...(pace !== 'normal' ? { speed: SPEED[pace] } : {}),
      }),
    });
    if (!r.ok) {
      const detail = (await r.text()).slice(0, 300);
      return json({ error: 'openai error', status: r.status, detail }, 502, origin);
    }

    // ⑤ 음성(mp3)을 그대로 앱에 전달
    return new Response(r.body, {
      headers: { 'Content-Type': 'audio/mpeg', 'Cache-Control': 'no-store', ...cors(origin) },
    });
  },
};
