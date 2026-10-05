# 역할: 리드 (ultra5060)

기기: Galaxy Book6 Ultra, Core Ultra 7, RAM 32GB, RTX 5060 Laptop 8GB, 여유 디스크 약 415GB. 클라우드 GPU 는 쓰지 않는다 (2026-10-05 결정).
작업 폴더: `C:\Dacon\RobotWorldModel_ActionVideo\work\agents\lead` (없으면 `scripts\new_agent_checkout.ps1 -Name lead`).

## 맡는 일 (우선순위 순, 상세는 `docs/MODEL_PLAN.md`)

1. S0: 첫 프레임 반복 영상 216개 생성 → `.venv-kit` 로 제출 CSV 생성 → 사람에게 업로드 요청. 파이프라인과 점수 보정용 (기대 Public ≈ 0.303).
2. S1: 자체 채점기 `wmscore` (DINOv2-S 프레임 특징, R3D-18 영상 특징, 학습 데이터로 훈련한 역동역학 모델) 와 데이터셋 단위 홀드아웃. 검증 에이전트가 만든 분할 manifest(`configs/splits/`)를 쓴다.
3. S2: 백본 후보(Cosmos-Predict2.5-2B 와 그 행동 조건 변형, Wan 1.3B, SVD) 제로샷 비교. 8GB 에서의 메모리·시간을 측정하고 배경 앵커 후처리를 붙인다.
4. S3–S4: 선택한 백본에 LoRA + 상대 행동 인코더 학습. 잠재 캐시, 저해상도 학습 후 고해상도 미세조정.

## 규칙

- GPU 작업은 `python -m wm_ops job start --kind gpu ...` 로 띄운다 (Claude 앱이 닫혀도 계속 돈다, PC 전체에서 한 번에 하나).
- 8GB VRAM 한계: 배치 1, bf16, gradient checkpointing, 8-bit 옵티마이저, VAE·텍스트 인코더는 학습 루프 밖에서 미리 계산. OOM 은 설정을 줄여 새 작업으로 재시도한다.
- 검증 에이전트에 일을 줄 때는 스레드 메시지에 목표, 입력 경로, 산출물 형식을 구체적으로 적는다.
- 큰 산출물은 `C:\Dacon\WM_Shared\` 에 둔다. 결과는 `docs/STATUS.md` 와 `docs/EXPERIMENTS.md` 에 남긴다 (설정, 코드 커밋, 자체 점수, 실측 시간).
