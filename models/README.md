# 글자 인식 모델 (우리 저장소에 직접 보관)

| 파일 | 내용 | 출처 |
|---|---|---|
| `korean_PP-OCRv5_mobile_rec_infer.onnx` | PaddleOCR PP-OCRv5 한국어 글자 읽기 모델 (한글+영어+숫자), 13MB | PaddleOCR (Apache-2.0) → ONNX 변환본: github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models `recognition/multi/korean/v5/` |
| `ppocrv5_korean_dict.txt` | 위 모델의 글자 사전 (첫 줄이 빈 줄 – 읽을 때 빈 줄은 건너뛸 것) | 같은 곳 |

sha256 (onnx): ee0dfde503d787c91fd0455daae2eb85311b4b5bdcddf85a54d8c1e0adc157de

남의 CDN·LFS 사용량 한도에 기대지 않으려고 여기에 둠. Cloudflare Pages 파일 한도(25MB) 안.
