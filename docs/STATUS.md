# STATUS

에이전트들이 함께 쓰는 현황판. 작업을 시작할 때 읽고, 끝날 때 갱신한다. 오래된 내용은 줄인다.

## 결정

- 2026-10-05: Ultra 한 대로 진행. 클라우드 GPU 는 빌리지 않고 RTX 5060 8GB 로 학습·추론 (`docs/MODEL_PLAN.md`).
- 2026-10-05: 에이전트는 이 프로젝트의 스레드 두 개(리드, 검증·데이터). 무인 `claude -p` 루프는 쓰지 않음.
- 2026-10-05: Pro 360 과 Syncthing 은 쓰지 않음. 작업 실행기는 창 없이(pythonw) 돈다.
- 2026-10-05 (JINWOO): S3 본 학습만 Colab(A100/H100)에서 한다. Ultra 는 데이터 준비(잠재 캐시)·채점·제출 파일. 전달은 Google Drive `MyDrive/dacon_wm`, 노트북 `colab/train_cosmos_ac.ipynb`.

## Now

- 운영: 리드는 `work\agents\lead`, 검증·데이터는 `work\agents\verifier` 복제본에서 일한다 (`docs/OPERATIONS.md`).
- 설치: Ultra 에 `.venv-ultra5060`, `.venv-kit`, 창 없는 작업 실행기(`WM-Jobs-ultra5060`) 설치 완료. Syncthing 과 Pro 360 구성은 삭제.

## Next

1. 완료: S0 첫 프레임 반복 제출 Public **0.3021**, Private 0.3341 (2026-10-05). 기대값 0.303 과 일치해 파이프라인 검증됨.
2. (검증) 완료: 데이터 인덱스 `C:\Dacon\WM_Shared\data_index\` (`tools/data_index/build_index.py`), 홀드아웃 `configs/splits/holdout_v1.json` (업로더 6명·데이터셋 12개·프레임 8.7%, 검증 창 192개 `holdout_v1_val_windows.csv`). 브랜치 `verify/data-index-split`.
3. 완료: 자체 채점기 `wmscore` (`python -m wmscore.score --pred <dir> --holdout C:\Dacon\WM_Shared\holdout_v1 --idm C:\Dacon\WM_Shared\idm\idm_v1.pt --out <json>`). 홀드아웃 192개는 eval 형식으로 `C:\Dacon\WM_Shared\holdout_v1`. 역동역학 모델 `idm_v1` (ResNet18+BiGRU, 128x208, train 데이터셋당 40 에피소드 캐시, 검증 MAE 0.485 z, 4천 스텝 이후 과적합). S0 홀드아웃 0.3185 (DINO 0.1031 / R3D 0.0784 / Action 0.6603) vs 리더보드 0.3021. DINO 는 제출킷처럼 518x518 입력. (검증) R3D 일치(0.0774), DINO 는 224 로 재서 다름 → 518 로 재확인 요청.
4. (리드) S2/S3: Cosmos-Predict2.5-2B `robot/action-cond` 를 diffusers `CosmosTransformer3DModel` 로 변환 (`python -m wmgen.cosmos_ac convert`, `C:\Dacon\WM_Shared\cosmos_ac`). 240x320, 17프레임(잠재 5) 생성. S3 는 새 6관절 행동 임베더(상대·차분·절대 18차원×4스텝) + LoRA r32 (`wmgen/train_cosmos_ac.py`), 학습은 Colab (`colab/train_cosmos_ac.ipynb`). 잠재 캐시 `wmgen/latent_cache.py` → `C:\Dacon\WM_Shared\latents_240x320`. Wan2.1-VACE-1.3B 제로샷은 비교용으로 대기열.

## Blocked

- Hugging Face 토큰: Cosmos-Predict2.5-2B 는 라이선스 동의(자동 승인)가 필요하다. 사람이 동의하고 Ultra 에서 `.venv-ultra5060\Scripts\hf.exe auth login` 으로 read 토큰을 저장하면 작업 실행기에서도 읽힌다 (환경 변수는 이미 떠 있는 실행기에 안 보임).

## Evidence

- 데이터 분석: `docs/DATA_ANALYSIS.md`, 재현 스크립트 `tools/analysis/`.
- 데이터 인덱스 (2026-10-05): 128 데이터셋, 11,132 에피소드, 1,025,666 프레임. parquet 행 수 = 영상 프레임 수 = episodes.jsonl 길이 (불일치·결손·NaN 0). 16프레임 미만 에피소드 34개.
- S0 (2026-10-05, 커밋 3cda924): 216개 생성 21초, 0번 프레임 평균 절대오차 ≤0.40 (yuv444p crf0). 제출킷 CSV 649행, 형식 일치. `.venv-kit` 에 `omegaconf` 가 없어 체크포인트 로드가 실패해서 설치 (`setup_env.ps1` 반영, 제출킷 코드는 그대로).
- 작업 대기열 시험: CPU·GPU 시험 작업 성공 (2026-10-05, `runs/*smoke*`).
