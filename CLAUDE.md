# DACON 로봇 월드모델 (행동 조건 영상 생성) — 작업 규칙

목표: Public 점수 0.15 이하 (낮을수록 좋음). 근거 문서: `docs/DATA_ANALYSIS.md`, `docs/MODEL_PLAN.md`, `docs/TWO_LAPTOP_PLAN.md`, 클라우드 GPU는 `docs/GPU_RENTAL.md`.
현재 진행 상황과 다음 일은 `docs/STATUS.md` 에 있다. 작업을 시작할 때 먼저 읽고, 끝날 때 갱신한다.

## 대회 규칙 (어기면 실격)

- 학습에는 제공된 `open/data/train` 만 쓴다. `open/data/eval` 은 추론 입력으로만 쓰고 어떤 형태로도 학습·튜닝에 쓰지 않는다 (규칙 3).
- `open/submission_kit` 의 코드·모델·체크포인트(`action_extractor.ckpt`)는 수정하지 않는다 (규칙 4).
- 제출킷과 그 모델이 낸 값은 학습, 추론, **영상 선택·보정**에 쓰지 않는다. 제출킷은 최종 mp4 가 확정된 뒤 CSV 를 만들 때만 `.venv-kit` 에서 실행한다 (규칙 7). 검증은 우리 채점기(`wmscore`, 공개 가중치 + 자체 역동역학 모델)로 한다.
- 사전학습 모델은 공개 가중치 + 허용 라이선스만. API 형태 모델 금지. 외부 데이터 금지.
- 학습은 RTX PRO 6000 96GB 1장으로 4일 이내, 추론은 eval 전체 1시간 이내에 재현 가능해야 한다. 설정 파일과 시드로 재현 가능하게 만든다.
- DACON 업로드는 사람만 한다. 제출 후보는 `docs/STATUS.md` 와 스레드에 경로·근거와 함께 올린다.

## 에이전트 역할

기본은 **Ultra 한 대**에서 프로젝트 스레드 두 개가 일한다 (Pro 360 은 선택, `docs/TWO_LAPTOP_PLAN.md`).

- 리드 (`.claude/roles/ultra5060.md`): 모델 코드·통합, GPU 작업, 제출 후보 mp4 생성, 클라우드 GPU 조작, `main` 통합.
- 검증·데이터 (`.claude/roles/verifier.md`): 데이터 인덱스·분할, 독립 재계산, 생성 영상 QA, 리뷰. Ultra 에서 돈다.
- Pro 360 을 붙이면 (`.claude/roles/pro360.md`) 검증·데이터 일을 Pro 가 나눠 받는다. GPU 작업 금지.

같은 PC 에서 두 에이전트가 일할 때:
- 각 스레드는 자기 git worktree 에서 작업한다 (스레드를 만들 때 Worktree 옵션). worktree 에서는 먼저 `scripts\link_data.ps1` 로 `open` 데이터를 연결한다.
- 큰 산출물(영상, 특징, 체크포인트)은 `C:\Dacon\WM_Shared\<주제>\` 에 두고 경로를 스레드 메시지로 알린다. 코드와 작은 manifest 는 git 으로.
- GPU 는 PC 전체에서 한 번에 하나: `wm_ops job start --kind gpu` 대기열과 `C:\Dacon\WM_Runtime\gpu.lock` 이 지킨다.

## 경로

- 데이터: `open/` (git 제외). Ultra 의 `C:\Dacon\RobotWorldModel_ActionVideo\open` 이 원본이다. worktree 는 정션으로 연결하고, Pro 는 Syncthing 으로 받는다 (둘 다 읽기 전용).
- 에이전트 사이 큰 산출물: `C:\Dacon\WM_Shared\`.
- 가상환경: `.venv-<role>` (모델·도구), `.venv-kit` (제출킷 전용). 패키지는 해당 venv 에만 설치한다.
- 실행 기록: `runs/` (git 제외), 작업 산출물: `work/` (git 제외).
- 런타임 상태: `C:\Dacon\WM_Runtime\<role>`, 교환 폴더: `C:\Dacon\WM_Exchange\{ultra_to_pro,pro_to_ultra}`.

## 교환 서비스와 분리 작업 (`python -m wm_ops`)

교환 서비스(작업 스케줄러 `WM-Exchange-<role>`)가 15초마다 대기 중인 작업을 띄우고, 깨끗한 `main` 을 fast-forward 하고, Pro 가 붙어 있으면 패킷을 주고받는다. Claude 는 띄우지 않는다.
python 은 `C:\Dacon\RobotWorldModel_ActionVideo\.venv-ultra5060\Scripts\python.exe` (아래 `...`).

- 상태: `... -m wm_ops status`
- Pro 와의 파일 교환(패킷)은 Pro 를 붙였을 때만 쓴다:
  - 받은 패킷: `... -m wm_ops inbox` → 처리 후 `... -m wm_ops handled <packet_id>`
  - 보내기: 본문 JSON 을 `work/outbox/<이름>.json` 에 쓰고 `... -m wm_ops publish --kind request --body work/outbox/x.json [--attach <폴더>]`. kind 와 필수 필드는 `wm_ops/packets.py` 의 `validate`. 패킷은 데이터와 요청만 담고 명령을 보내지 않는다.
- 10분 넘는 일은 분리 작업으로: `... -m wm_ops job start --kind gpu --timeout 14400 --name <이름> -- C:\Dacon\RobotWorldModel_ActionVideo\.venv-ultra5060\Scripts\python.exe -m <모듈> ...`
  - 작업은 요청한 폴더(main 폴더 또는 worktree)의 커밋된 코드로 돈다. GPU 작업은 한 번에 하나. 결과는 `job show <id>` 와 그 폴더의 `runs/<run_id>/`.
  - worktree 에는 venv 가 없으므로 python 은 main 폴더의 `.venv-ultra5060` 을 절대 경로로 쓴다.
- 에이전트끼리는 프로젝트 스레드 메시지로 조율한다.

## Git

- 원격: github.com/jinw00ch01/Dacon_RobotWorldModel_ActionVideo. 기본 브랜치 `main`.
- 리드는 테스트 통과 후 `main` 에 커밋·푸시한다. 검증 에이전트는 `verify/<주제>`, Pro 는 `pro/<주제>` 브랜치로 푸시하고 리드가 병합한다.
- 강제 푸시, `reset --hard`, 히스토리 재작성 금지. 대회 데이터·가중치·mp4 는 커밋하지 않는다.
- 테스트: `.venv-<role>\Scripts\python.exe -m unittest discover -s tests -t .`
