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

## 노트북 역할 (`configs/local-node.json` 의 `role`)

- `ultra5060` (Galaxy Book6 Ultra, RTX 5060 8GB): 리드. 모델 코드·통합, GPU 작업, 제출 후보 mp4 생성, 클라우드 GPU 조작, `main` 통합.
- `pro360` (Galaxy Book3 Pro 360, CPU 전용): 데이터 분할·인덱스, 자체 채점기 CPU 재계산, 생성 영상 QA, 코드 리뷰. GPU 작업 금지.
- 역할 상세: `.claude/roles/<role>.md`.

## 경로

- 데이터: `open/` (git 제외). Ultra 가 원본이고 Pro 는 Syncthing 으로 받는다 (Pro 쪽은 읽기 전용).
- 가상환경: `.venv-<role>` (모델·도구), `.venv-kit` (제출킷 전용). 패키지는 해당 venv 에만 설치한다.
- 실행 기록: `runs/` (git 제외), 작업 산출물: `work/` (git 제외).
- 런타임 상태: `C:\Dacon\WM_Runtime\<role>`, 교환 폴더: `C:\Dacon\WM_Exchange\{ultra_to_pro,pro_to_ultra}`.

## 두 노트북 사이 협업 (`python -m wm_ops`)

교환 서비스(작업 스케줄러 `WM-Exchange-<role>`)가 15초마다 패킷을 가져오고, 대기 중인 작업을 띄우고, 깨끗한 `main` 을 fast-forward 한다. Claude 는 띄우지 않는다.

- 상태: `.venv-<role>\Scripts\python.exe -m wm_ops status`
- 받은 패킷: `... -m wm_ops inbox` → 처리 후 `... -m wm_ops handled <packet_id>`
- 패킷 보내기: 본문 JSON 을 `work/outbox/<이름>.json` 에 쓰고 `... -m wm_ops publish --kind request --body work/outbox/x.json [--attach <폴더>]`
  - kind: `spec`/`decision`(Ultra), `review`/`qa`(Pro), `result`/`data`/`request`(둘 다). 필수 필드는 `wm_ops/packets.py` 의 `validate` 참고.
  - 패킷은 데이터와 요청만 담는다. 상대가 실행할 명령을 보내지 않는다.
- 10분 넘는 일은 분리 작업으로: `... -m wm_ops job start --kind gpu --timeout 14400 --name <이름> -- .venv-ultra5060\Scripts\python.exe -m <모듈> ...`
  - GPU 작업은 커밋된 코드에서만, 한 번에 하나. 결과는 `job show <id>` 와 `runs/<run_id>/`.
- 같은 시각에 두 노트북에 Claude 세션이 떠 있으면 프로젝트 스레드끼리 메시지로 조율하고, 파일은 패킷으로 넘긴다.

## Git

- 원격: github.com/jinw00ch01/Dacon_RobotWorldModel_ActionVideo. 기본 브랜치 `main`.
- Ultra 는 테스트 통과 후 `main` 에 커밋·푸시한다. Pro 는 `pro/<주제>` 브랜치로 푸시하고 Ultra 가 병합한다 (예외: `configs/nodes.json` 의 자기 장치 ID 등록).
- 강제 푸시, `reset --hard`, 히스토리 재작성 금지. 대회 데이터·가중치·mp4 는 커밋하지 않는다.
- 테스트: `.venv-<role>\Scripts\python.exe -m unittest discover -s tests -t .`
