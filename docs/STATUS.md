# STATUS

두 노트북의 Claude 세션이 함께 쓰는 현황판. 작업을 시작할 때 읽고, 끝날 때 갱신한다. 오래된 내용은 줄인다.

## Now

- 설치: Ultra 에 `.venv-ultra5060`, `.venv-kit`, 교환 서비스(`WM-Exchange-ultra5060`) 설치. Pro 360 은 그 노트북의 Claude 세션이 `docs/TWO_LAPTOP_PLAN.md` 의 "Pro 360 설치" 를 실행하면 끝난다.
- 클라우드 GPU: RTX PRO 6000 대여 가능 확인 (`docs/GPU_RENTAL.md`). 사람의 계정·충전·API 키 대기.

## Next

1. (Ultra) S0 첫 프레임 반복 제출 파일 만들기 → 사람에게 업로드 요청.
2. (Pro) 데이터 인덱스와 데이터셋 단위 홀드아웃 분할 manifest → `data` 패킷.
3. (Ultra) 자체 채점기 `wmscore` + 홀드아웃 보정.
4. (Ultra) Cosmos-Predict2.5-2B 8GB 실현성 시험.

## Blocked

- 클라우드 학습: RunPod 계정·충전·`RUNPOD_API_KEY`, Hugging Face `HF_TOKEN` (사람).
- Pro 360 설치: Pro 에 이 프로젝트의 Claude 세션이 필요 (사람이 Claude 앱에서 폴더 허용).

## Evidence

- 데이터 분석: `docs/DATA_ANALYSIS.md`, 재현 스크립트 `tools/analysis/`.
