# EXPERIMENTS

제출과 주요 실험 기록. 한 줄에 하나, 최신이 아래.

| 날짜 | ID | 내용 (바꾼 것 하나) | 코드 커밋 | 자체 점수 (DINO / R3D / Action / 합) | Public | 추론 시간 | 비고 |
|---|---|---|---|---|---|---|---|
| 2026-10-05 | S0 | 첫 프레임 반복 (yuv444p crf0) | 3cda924 | 0.1031 / 0.0784 / 0.6603 / **0.3185** (wmscore, idm_v1) | **0.3021** (Private 0.3341) | 21초 (CPU) | `C:\Dacon\WM_Shared\s0_first_frame` |
| 2026-10-05 | S2-cosmos-diag | Cosmos-Predict2.5-2B action-cond 변환 진단 (영 행동, 256x320, 25스텝) | 454c18a | - | - | 13.6초/샘플 (35스텝, 240x320), 최대 5.2GB | 조건 프레임에 작은 timestep(1e-4)을 주면 화면이 깨짐(|Δ| 53~82). 모든 잠재 프레임에 같은 timestep 이 맞음(|Δ| 8~14). fps=4 RoPE 가 가장 안정(|Δ| 4~11). `C:\Dacon\WM_Shared\cosmos_ac\diag` |
| 2026-10-05 | S2-wan-vace | Wan2.1-VACE-1.3B 제로샷 (480x640, 17프레임, 20스텝, CFG 5) | f0f57b1 | - | - | 샘플당 342초, 최대 10.7GB (8GB 초과로 공유 메모리 사용) | 2개만 생성 후 중단. 8GB 에서 비실용적이라 후보 제외. Cosmos action-cond 로 진행 |
