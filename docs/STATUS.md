# STATUS

에이전트들이 함께 쓰는 현황판. 작업을 시작할 때 읽고, 끝날 때 갱신한다. 오래된 내용은 줄인다.

## 결정

- 2026-10-05: Ultra 한 대로 진행. 클라우드 GPU 는 빌리지 않고 RTX 5060 8GB 로 학습·추론 (`docs/MODEL_PLAN.md`).
- 2026-10-05: 에이전트는 이 프로젝트의 스레드 두 개(리드, 검증·데이터). 무인 `claude -p` 루프는 쓰지 않음.
- 2026-10-05: Pro 360 과 Syncthing 은 쓰지 않음. 작업 실행기는 창 없이(pythonw) 돈다.
- 2026-10-05 (JINWOO): S3 본 학습만 Colab(A100/H100)에서 한다. Ultra 는 데이터 준비(잠재 캐시)·채점·제출 파일. 전달은 Google Drive `MyDrive/dacon_wm`, 노트북 `colab/train_cosmos_ac.ipynb`. Claude 의 안전장치가 대회 데이터(파생 잠재 포함)를 Drive 로 옮기는 걸 막으므로, 잠재 캐시와 행동 통계 json 의 Drive 복사는 사람이 직접 한다 (`robocopy C:\Dacon\WM_Shared\latents_240x320 "G:\내 드라이브\dacon_wm\latents_240x320" /E`). 변환한 Cosmos 가중치는 이미 Drive 에 있음.

- 2026-10-06 (JINWOO): 이 PC 의 다른 프로젝트(microstructure-hardness-prediction)와 GPU 를 번갈아 쓴다. 그쪽은 GPU 명령을 `wm_ops/gpu_turn.py` 로 감싸 실행하고, 작업 실행기는 그동안 우리 GPU 작업을 일시정지했다가 끝나면 재개한다 (`docs/OPERATIONS.md`). 우리 작업에는 바꿀 것이 없다.

## Now

- 운영: 리드는 `work\agents\lead`, 검증·데이터는 `work\agents\verifier` 복제본에서 일한다 (`docs/OPERATIONS.md`).
- 설치: Ultra 에 `.venv-ultra5060`, `.venv-kit`, 창 없는 작업 실행기(`WM-Jobs-ultra5060`) 설치 완료. Syncthing 과 Pro 360 구성은 삭제.

## Next

1. 완료: S0 첫 프레임 반복 제출 Public **0.3021**, Private 0.3341 (2026-10-05). 기대값 0.303 과 일치해 파이프라인 검증됨.
2. (검증) 완료: 데이터 인덱스 `C:\Dacon\WM_Shared\data_index\` (`tools/data_index/build_index.py`), 홀드아웃 `configs/splits/holdout_v1.json` (업로더 6명·데이터셋 12개·프레임 8.7%, 검증 창 192개 `holdout_v1_val_windows.csv`). 브랜치 `verify/data-index-split`.
3. 완료: 자체 채점기 `wmscore` (`python -m wmscore.score --pred <dir> --holdout C:\Dacon\WM_Shared\holdout_v1 --idm C:\Dacon\WM_Shared\idm\idm_v1.pt --out <json>`). 홀드아웃 192개는 eval 형식으로 `C:\Dacon\WM_Shared\holdout_v1`. 역동역학 모델 `idm_v1` (ResNet18+BiGRU, 128x208, train 데이터셋당 40 에피소드 캐시, 검증 MAE 0.485 z, 4천 스텝 이후 과적합). S0 홀드아웃 0.3185 (DINO 0.1031 / R3D 0.0784 / Action 0.6603) vs 리더보드 0.3021. DINO 는 제출킷처럼 518x518 입력. (검증) R3D 일치(0.0774), DINO 는 224 로 재서 다름 → 518 로 재확인 요청.
4. (리드) S2/S3: Cosmos-Predict2.5-2B `robot/action-cond` 를 diffusers `CosmosTransformer3DModel` 로 변환 (`python -m wmgen.cosmos_ac convert`, `C:\Dacon\WM_Shared\cosmos_ac`). 240x320, 17프레임(잠재 5) 생성. S3 는 새 6관절 행동 임베더(상대·차분·절대 18차원×4스텝) + LoRA r32 (`wmgen/train_cosmos_ac.py`), 학습은 Colab (`colab/train_cosmos_ac.ipynb`). 잠재 캐시 `wmgen/latent_cache.py` → `C:\Dacon\WM_Shared\latents_240x320`. Wan2.1-VACE-1.3B 제로샷은 비교용으로 대기열.
5. (리드, 2026-10-07 15:30 KST) 제출 후보 2 완성: `C:\Dacon\WM_Shared\submissions\v2_s8000_g3\submission_v2_s8000_g3_anchor_t20.csv` (v2 8k + guidance 3 + 배경 앵커 t20, sub64 자체 0.2755 = 첫 프레임 반복과 동률). JINWOO 업로드 여부 결정 대기. v3(12k)는 v2 보다 나빠 보류 (최고 0.296). 다음 개선: DINO(팔 영역 화질) 병목. 대기·실행 중인 리드 작업 없음.

## Blocked

- Hugging Face 토큰: Cosmos-Predict2.5-2B 는 라이선스 동의(자동 승인)가 필요하다. 사람이 동의하고 Ultra 에서 `.venv-ultra5060\Scripts\hf.exe auth login` 으로 read 토큰을 저장하면 작업 실행기에서도 읽힌다 (환경 변수는 이미 떠 있는 실행기에 안 보임).

## Evidence

- 데이터 분석: `docs/DATA_ANALYSIS.md`, 재현 스크립트 `tools/analysis/`.
- 데이터 인덱스 (2026-10-05): 128 데이터셋, 11,132 에피소드, 1,025,666 프레임. parquet 행 수 = 영상 프레임 수 = episodes.jsonl 길이 (불일치·결손·NaN 0). 16프레임 미만 에피소드 34개.
- S0 (2026-10-05, 커밋 3cda924): 216개 생성 21초, 0번 프레임 평균 절대오차 ≤0.40 (yuv444p crf0). 제출킷 CSV 649행, 형식 일치. `.venv-kit` 에 `omegaconf` 가 없어 체크포인트 로드가 실패해서 설치 (`setup_env.ps1` 반영, 제출킷 코드는 그대로).
- 작업 대기열 시험: CPU·GPU 시험 작업 성공 (2026-10-05, `runs/*smoke*`).
- GPU 공유 실측 (2026-10-06 13:46 KST, 커밋 d251b53): `gpu_turn.py` 요청 후 11초 만에 허가, latents-resume 의 작업 프로세스 3개 일시정지, 명령이 끝난 뒤 다음 폴링에 재개. 이때는 잠재 캐시가 배터리로 AC 대기 중이어서 13:53 KST 에 계산 중인 잠재 캐시로 다시 쟀다: 25초 차례 요청 후 12초 뒤 GPU 사용률 100%→0%, 차례 동안 0%, 끝나고 4초 뒤 100%. 멈췄던 데이터셋 `pietroom__actualeasytask.pt` 는 정상 저장(240 클립, 값 유한, 구조 동일). 실행 중인 실행기 루프는 예전 코드라 GPU 차례 기록이 `runner.log` 에 남지 않는다(다음 재시작부터 남음). 상태는 `C:\Dacon\WM_Runtime\ultra5060\ops\gpu-share.json` 과 `C:\Dacon\WM_Runtime\gpu.granted`.
