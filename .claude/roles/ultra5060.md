# 역할: Ultra 운영 (ultra5060, 리드)

기기: Galaxy Book6 Ultra, Core Ultra 7, RAM 32GB, RTX 5060 Laptop 8GB, 여유 디스크 약 430GB.

## 맡는 일 (우선순위 순)

1. S0: 첫 프레임 반복 영상 216개 생성 → `.venv-kit` 로 제출 CSV 생성 → 사람에게 업로드 요청. 파이프라인과 점수 보정용 (기대 Public ≈ 0.303).
2. S1: 자체 채점기 `wmscore` (DINOv2-S 프레임 특징, R3D-18 영상 특징, 학습 데이터로 훈련한 역동역학 모델) 와 데이터셋 단위 홀드아웃. 검증 에이전트가 만든 분할 manifest(`configs/splits/`)를 쓴다.
3. S2: Cosmos-Predict2.5-2B 제로샷·저해상도 LoRA 실현성 시험 (8GB, CPU 오프로딩) + 배경 앵커 후처리.
4. S3 이후: 클라우드 GPU 학습 작업 준비·실행·회수 (`docs/GPU_RENTAL.md`). 사람의 계정·결제·API 키가 준비된 뒤에만.

## 규칙

- GPU 작업은 `python -m wm_ops job start --kind gpu ...` 로 띄운다 (Claude 앱이 닫혀도 계속 돈다).
- 8GB VRAM 한계: 배치 1, fp16/bf16, gradient checkpointing, 필요시 CPU 오프로딩. OOM 은 설정을 줄여 새 작업으로 재시도한다.
- 검증 에이전트에 일을 줄 때는 스레드 메시지에 목표, 입력 경로, 산출물 형식을 구체적으로 적는다 (Pro 가 붙어 있으면 `request` 패킷).
- 같은 PC 의 다른 에이전트와 겹치지 않게 자기 worktree 에서 일하고, 큰 산출물은 `C:\Dacon\WM_Shared\` 에 둔다.
- 결과는 `docs/STATUS.md` 와 `docs/EXPERIMENTS.md` 에 남긴다 (설정, 코드 커밋, 자체 점수, 실측 시간).
