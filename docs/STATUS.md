# STATUS

두 노트북의 Claude 세션이 함께 쓰는 현황판. 작업을 시작할 때 읽고, 끝날 때 갱신한다. 오래된 내용은 줄인다.

## Now

- 운영: Ultra 단독이 기본. 리드 스레드와 검증·데이터 스레드가 각자 worktree 에서 일한다 (`docs/TWO_LAPTOP_PLAN.md` 0절).
- 설치: Ultra 에 `.venv-ultra5060`, `.venv-kit`, 교환 서비스(`WM-Exchange-ultra5060`) 설치 완료. Pro 360 은 선택: 붙여넣기 한 줄로 설치·페어링 (3절).
- 클라우드 GPU: RTX PRO 6000 대여 가능 확인 (`docs/GPU_RENTAL.md`). 사람의 계정·충전·API 키 대기.

## Next

1. (리드) S0 첫 프레임 반복 제출 파일 만들기 → 사람에게 업로드 요청.
2. (검증) 데이터 인덱스와 데이터셋 단위 홀드아웃 분할 manifest → `configs/splits/`.
3. (리드) 자체 채점기 `wmscore` + 홀드아웃 보정. (검증) 독립 재계산.
4. (리드) Cosmos-Predict2.5-2B 8GB 실현성 시험.

## Blocked

- 클라우드 학습: RunPod 계정·충전·`RUNPOD_API_KEY`, Hugging Face `HF_TOKEN` (사람).
- (선택) Pro 360: 사람이 Pro 의 PowerShell 에 참여 코드가 든 설치 명령을 붙여넣기 (Remote Control 은 Pro 에 닿지 않음).

## Evidence

- 데이터 분석: `docs/DATA_ANALYSIS.md`, 재현 스크립트 `tools/analysis/`.
