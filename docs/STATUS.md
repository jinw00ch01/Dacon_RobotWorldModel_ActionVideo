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
<<<<<<< HEAD
5. (리드, 2026-10-08 19:55 KST, 재시작 대비) 대기·실행 중인 리드 작업 없음. 제출 3 = Public 0.2273 (현재 최고). v2_long2 채점 완료, 최고는 22k (sub64 0.2528, 16k 0.2602 대비 -0.0074 → 보정상 Public 약 0.20 기대, 추정). 체크포인트 사본: `C:\Dacon\WM_Shared\cosmos_ac
uns
2_long2dapter_0{18,20,22,24}000.pt`. 재개 후 할 일: JINWOO 가 원하면 제출 4 = `wmgen.submit_best --out C:\Dacon\WM_Shared\submissions\sub4 --guidance 3 --threshold 20 --candidate v2long2_s22000=<22k adapter>=<sub64 json>` (약 1.7시간). 보정(검증, 3점 추정): 리더보드는 홀드아웃 변화의 약 3.5배 → Public 0.15 에는 홀드아웃 약 0.235~0.24 필요.

- (검증, 2026-10-09 15:30 KST) 제출 4 형식 QA 완료. 진행 중인 검증 작업 없음. 다음: 제출 4 리더보드 결과가 나오면 홀드아웃과 대조, 새 출력이 나오면 `qa_when_ready.py`.
erify_scores\sub64_*_idm_v2.csv`. idm-v2 가 실패하면 이 작업들도 실패하므로 다시 넣는다. 결과가 나오면 행동 항목 폭(GT 바닥 / 정지 / 베이스라인)을 리드에게 보고. 그다음 리드 어댑터 예측 콘택트 시트 QA (`tools/verify/qa_sheet.py`).
=======
5. (리드, 2026-10-09 15:30 KST) 제출 4 완성: `C:\Dacon\WM_Shared\submissions\sub4\submission_v2long2_s22000_g3_anchor_t20.csv` (v2_long2 22k + guidance 3 + 앵커 t20, sub64 0.2528). JINWOO 업로드 대기. 제출 3 = Public 0.2273 (현재 최고). GPU 는 microstructure 에 양도, 대기·실행 중인 리드 작업 없음.
uns
2_long2dapter_0{18,20,22,24}000.pt`. 재개 후 할 일: JINWOO 가 원하면 제출 4 = `wmgen.submit_best --out C:\Dacon\WM_Shared\submissions\sub4 --guidance 3 --threshold 20 --candidate v2long2_s22000=<22k adapter>=<sub64 json>` (약 1.7시간). 보정(검증, 3점 추정): 리더보드는 홀드아웃 변화의 약 3.5배 → Public 0.15 에는 홀드아웃 약 0.235~0.24 필요.
>>>>>>> origin/main

## Blocked

- Hugging Face 토큰: Cosmos-Predict2.5-2B 는 라이선스 동의(자동 승인)가 필요하다. 사람이 동의하고 Ultra 에서 `.venv-ultra5060\Scripts\hf.exe auth login` 으로 read 토큰을 저장하면 작업 실행기에서도 읽힌다 (환경 변수는 이미 떠 있는 실행기에 안 보임).

## Evidence

- 데이터 분석: `docs/DATA_ANALYSIS.md`, 재현 스크립트 `tools/analysis/`.
- 데이터 인덱스 (2026-10-05): 128 데이터셋, 11,132 에피소드, 1,025,666 프레임. parquet 행 수 = 영상 프레임 수 = episodes.jsonl 길이 (불일치·결손·NaN 0). 16프레임 미만 에피소드 34개.
- 검증 도구 (2026-10-05, `tools/verify/`): `feature_scores.py` (DINO·R3D 독립 재계산), `qa_sheet.py` (16프레임·프레임0 PSNR·배경 drift·움직임 방향 + 콘택트 시트). 홀드아웃 192창 첫 프레임 반복 보정값 (CPU): DINO 0.1157, R3D 0.0799. GT 자기 자신 0/0.
- 독립 재계산 S0 (2026-10-05, 커밋 이 브랜치 `tools/verify/feature_scores.py`, DINO 518): 0.3178 = DINO 0.1017, R3D 0.0774, Action 0.6603. 리드 wmscore 0.3185 와 차이 0.0007, 샘플별 상관 DINO 0.998, R3D 0.981, Action 1.000.
- 공식 베이스라인 보정 (2026-10-05, holdout_v1 64창 `C:\Dacon\WM_Shared\holdout_v1_sub64`, `tools/verify/run_baseline.py`, 생성 18분): 우리 채점 0.5124 (DINO 0.547, R3D 0.233, Action 0.696) vs 리더보드 0.517. 같은 64창 S0 0.3230 vs 리더보드 0.302. 순서와 간격(0.19 vs 0.215)이 리더보드와 맞음.
- idm_v2 행동 항목 폭 (2026-10-06, 64창, 독립 채점): GT 0.184 / 정지(S0) 0.560 / 베이스라인 0.582 (idm_v1 은 약 0.49 / 0.678 / 0.696). 총점 S0 0.276, 베이스라인 0.467 (리더보드 0.302, 0.517): 순서와 간격(0.19 vs 0.215)은 유지, 절대값은 v1 보다 0.03–0.05 낮음. 리드 wmscore 와 샘플별 행동 항목 상관 1.000. idm_v2 학습 목록에 홀드아웃 창 에피소드 0개, IDM train 목록에 val 데이터셋 에피소드 0개 (val 데이터셋은 valtrain 376개만) 확인 (`C:\Dacon\WM_Shared\idm_cache\manifest.json`).
- 어댑터 v2_full 2000스텝 QA (2026-10-06, 64창, `tools/verify/action_direction.py`): 0번 프레임 = 입력 (PSNR ≥52dB), 16프레임. 관절별 움직임 상관 0.11–0.52 (GT 영상 0.70–0.98), 음수 관절 없음 → 부호·순서 오류 아님. t=15 움직임 크기가 목표의 0.11–0.40 → 행동 조건이 약함. 배경 고정판 배경 drift 0.04 (GT 1.5). 시트 `C:/Dacon/WM_Shared/verify_qa/sub64_v2full_s2000*`.
- 어댑터 4000스텝 QA·독립 재채점 (2026-10-06, idm_v2): holdout 64창 g0 0.327 (DINO 0.253, Action 0.578), g3 0.305 (DINO 0.280, Action 0.502); 리드 0.326/0.303 과 일치. 관절별 움직임 상관/크기비: holdout g0 0.22–0.34 / 0.21–0.49, g3 0.27–0.67 / 0.35–0.79, 학습 프로브 32창 s4000 0.20–0.72 / 0.33–0.64 (S0 ≈0). → 학습 클립에서도 움직임이 목표의 절반 수준이라 조건 강도 문제가 크고, 일반화 손실이 더해진다. 가이던스가 행동 항목을 가장 많이 줄이지만 DINO 가 스텝·가이던스와 함께 나빠져(0.17→0.25→0.28) 총점은 아직 S0(0.276)보다 나쁘다.
- 어댑터 후속 (2026-10-06, CPU 독립 채점, idm_v2): s4000_g3_anchor 0.2999 (DINO 0.261, Action 0.501, 배경 drift 0.42), s6000_g0 0.2942 (DINO 0.220, Action 0.524). s8000_g3: 리드 0.2818 (DINO 0.283, Action 0.443); 내 행동 MAE 0.4429 일치. 관절 상관 pan 0.79, lift 0.74, elbow 0.69, wflex 0.56, wroll 0.19, grip 0.53, 크기비 0.40–0.90 (지금까지 최고). 배경 drift 3.4 (GT 1.5) → 배경 고정을 붙이면 DINO 가 나아질 여지.
- s8000_g3 배경 고정 (2026-10-06, CPU 독립 채점, idm_v2): 고정 없음 0.2828 (리드 0.2818), t12 0.2785 (0.2771), t20 0.2770 (0.2755). 우리 값이 일관되게 약 0.0015 높음 (DINO 경로 차이). t20: DINO 0.263, R3D 0.070, Action 0.443, 배경 drift 0.24. 행동 방향·크기는 고정 전과 같음. 최고점이 S0(0.276)와 비김 → DINO(0.26 vs S0 0.10)가 남은 병목.
- v3 (조건 강화) QA·CPU 독립 채점 (2026-10-07, idm_v2, 64창; 괄호는 리드): s2000_g0 0.3470 (0.3457), s2000_g3 0.3763 (0.3739), s6000_g3 0.3085 (0.3069), s12000_g0 0.3152 (0.3143), s12000_g3 0.2971 (0.2962). 모두 16프레임, 0번 프레임 = 입력. s12000_g3: DINO 0.292, Action 0.472, 관절 상관 0.26–0.75, 크기비 0.46–0.86 → 움직임은 v2 8k_g3 와 비슷, DINO 가 더 나빠 v2 최고(0.2755)를 넘지 못함.
- 제출 2 (v2 8k+g3+anchor t20, eval 216) 형식 QA (2026-10-07): 216개 16프레임, 0번 프레임 PSNR ≥52dB, 4개(sample 8, 85, 86, 87)는 배경 고정 후 정지. 생성 25.4초/개, 216개 93분 (RTX 5060 노트북). 리더보드 public 0.2816 / private 0.2604 vs S0 0.3021 / 0.3341. 같은 두 영상의 홀드아웃 64창 점수: idm_v2 0.2770 vs 0.2757 (비김), idm_v1 0.3364 vs 0.3230 (S0 가 더 좋음). → 우리 채점기는 두 IDM 모두 움직이는 모델을 리더보드보다 박하게 본다. 모델끼리 비교에는 쓰되, S0 와의 절대 비교는 리더보드가 더 후하다는 점을 감안할 것.
- v2_long 16000스텝+g3+anchor t20 (2026-10-07): 홀드아웃 64창 CPU 독립 0.2611 (리드 0.2602), DINO 0.223, Action 0.438, 관절 상관 0.27–0.78, 크기비 0.38–0.80 (지금까지 최고). 제출 3 (eval 216, `C:/Dacon/WM_Shared/submissions/sub3/submission_v2long_s16000_g3_anchor_t20.csv`, 648행 = 216×3) 형식 QA 통과: raw·anchor 모두 16프레임, 0번 프레임 PSNR ≥52dB, anchor 후 정지 4개(sample 8, 85, 86, 87). 생성 24.8초/개.
- 리더보드 대조 (2026-10-07): 제출 3 public 0.2273 / private 0.2258. 세 점 (S0, 제출 2, 제출 3): 리더보드 public 0.302 / 0.282 / 0.227, 홀드아웃 idm_v2 64창 0.276 / 0.2755 / 0.2602. 순서는 같지만 리더보드 개선폭이 홀드아웃의 약 3.5배 (제출 2→3: −0.054 vs −0.015). → 우리 채점기는 방향은 맞고 폭을 작게 본다. 홀드아웃 0.01 개선이 리더보드 약 0.03 에 해당한다고 보면 목표 0.15 까지 홀드아웃으로 약 0.02–0.025 더 필요 (추정).
- 16k 설정 비교·v2_long2 (2026-10-08, CPU 독립, idm_v2, anchor t20; 괄호는 리드): 16k g2 0.2653 (0.2645), g4 0.2627 (0.2616), g3 50스텝 0.2615 (0.2605); v2_long2 18k 0.2646 (0.2632), 20k 0.2583 (0.2576), 22k 0.2540 (0.2528), 24k 0.2560 (0.2547). 22k 가 최고: Action 0.408 (최저), DINO 0.238, 관절 상관 pan 0.80, lift 0.81, elbow 0.74, wflex 0.54, wroll 0.20, grip 0.57. 모두 16프레임, 0번 프레임 = 입력.
- 제출 4 (v2_long2 22k+g3+anchor t20, `C:/Dacon/WM_Shared/submissions/sub4/submission_v2long2_s22000_g3_anchor_t20.csv`, 648행 = 216×3) 형식 QA 통과 (2026-10-09): raw·anchor 모두 16프레임, 0번 프레임 PSNR ≥52dB, anchor 후 정지 4개(sample 8, 85, 86, 87, 이전 제출과 같음).
- S0 (2026-10-05, 커밋 3cda924): 216개 생성 21초, 0번 프레임 평균 절대오차 ≤0.40 (yuv444p crf0). 제출킷 CSV 649행, 형식 일치. `.venv-kit` 에 `omegaconf` 가 없어 체크포인트 로드가 실패해서 설치 (`setup_env.ps1` 반영, 제출킷 코드는 그대로).
- 작업 대기열 시험: CPU·GPU 시험 작업 성공 (2026-10-05, `runs/*smoke*`).
- GPU 공유 실측 (2026-10-06 13:46 KST, 커밋 d251b53): `gpu_turn.py` 요청 후 11초 만에 허가, latents-resume 의 작업 프로세스 3개 일시정지, 명령이 끝난 뒤 다음 폴링에 재개. 이때는 잠재 캐시가 배터리로 AC 대기 중이어서 13:53 KST 에 계산 중인 잠재 캐시로 다시 쟀다: 25초 차례 요청 후 12초 뒤 GPU 사용률 100%→0%, 차례 동안 0%, 끝나고 4초 뒤 100%. 멈췄던 데이터셋 `pietroom__actualeasytask.pt` 는 정상 저장(240 클립, 값 유한, 구조 동일). 실행 중인 실행기 루프는 예전 코드라 GPU 차례 기록이 `runner.log` 에 남지 않는다(다음 재시작부터 남음). 상태는 `C:\Dacon\WM_Runtime\ultra5060\ops\gpu-share.json` 과 `C:\Dacon\WM_Runtime\gpu.granted`.
