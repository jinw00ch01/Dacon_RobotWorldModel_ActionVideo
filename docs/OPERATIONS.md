# 멀티 에이전트 운영 (Ultra 한 대)

2026-10-05 결정: 프로젝트는 Galaxy Book6 Ultra 한 대로 진행하고, 클라우드 GPU 는 빌리지 않으며, Pro 360 과 Syncthing 은 쓰지 않는다.

## 구조

```
     이 프로젝트 (claude.ai Projects) — 코디네이터 Claude 가 스레드 배정
            │                                  │
   ┌────────┴─────────┐   스레드 메시지   ┌────────┴───────────┐
   │ 리드 스레드       │◄──────────────►│ 검증·데이터 스레드   │   둘 다 Ultra 의 Claude 앱
   │ work\agents\lead  │                 │ work\agents\verifier │   (각자 git 복제본)
   └────────┬─────────┘                 └────────┬───────────┘
            │  open 정션 → C:\Dacon\RobotWorldModel_ActionVideo\open (원본 데이터, 읽기 전용)
            │  큰 산출물 → C:\Dacon\WM_Shared\
            └── python -m wm_ops job start ──► WM-Jobs-ultra5060 (작업 스케줄러, 창 없음) ── RTX 5060 작업 한 번에 하나
```

- **에이전트** = 이 프로젝트의 스레드 세션 두 개. 리드(`.claude/roles/ultra5060.md`)와 검증·데이터(`.claude/roles/verifier.md`). 서로 스레드 메시지로 조율한다.
- **작업 폴더**: 에이전트마다 자기 복제본 `work\agents\<이름>` (`scripts\new_agent_checkout.ps1 -Name <이름>`, 데이터 정션 포함). main 폴더 `C:\Dacon\RobotWorldModel_ActionVideo` 는 venv·데이터·설정이 있는 곳이고, 작업 실행기가 `main` 을 따라가도록 깨끗하게 둔다.
- **작업 실행기** `WM-Jobs-ultra5060`: 로그온 시 시작, 죽으면 1분 뒤 재시작. `pythonw -m wm_ops serve` 가 15초마다
  1. 죽은 작업을 `lost` 로 표시하고,
  2. 대기 중인 작업을 띄우고 (GPU 작업은 PC 전체에서 한 번에 하나, `C:\Dacon\WM_Runtime\gpu.lock`),
  3. 깨끗한 main 폴더를 5분마다 `main` 으로 fast-forward 한다.
  작업은 요청한 복제본의 커밋된 코드로 돌고, 끝나면 다음 대기 작업을 바로 띄운다. Claude 는 띄우지 않는다.
- **창이 뜨지 않게**: 실행기는 콘솔이 없는 pythonw 로 돌고, 자식 프로세스는 모두 `CREATE_NO_WINDOW` 로 띄운다. Windows 11 은 새 콘솔을 Windows Terminal 창으로 넘기기 때문에, 이전 Syncthing 감시 프로세스(숨김 PowerShell)와 `DETACHED_PROCESS` 로 띄운 작업이 창을 띄웠다.

## 설치 (Ultra, 완료)

```
scripts\setup_env.ps1 -Role ultra5060 -Kit     # .venv-ultra5060 (torch 2.8.0+cu128), .venv-kit (제출킷 고정 버전)
scripts\setup_node.ps1                         # configs/local-node.json, 작업 WM-Jobs-ultra5060 (창 없음)
scripts\new_agent_checkout.ps1 -Name lead      # 에이전트 복제본 (verifier 도 같은 방식)
```

- 상태: `.venv-ultra5060\Scripts\python.exe -m wm_ops status`
- 멈춤/재개: `... -m wm_ops stop` / `... -m wm_ops resume` (실행 중인 작업은 계속 돈다)

## 기록

- 2026-10-05: 두 노트북(Pro 360) 구성을 위해 Syncthing 교환·참여 코드 페어링을 만들었다가, Ultra 단독 결정에 따라 모두 지웠다 (git 기록에 남아 있음). Pro 에 설치된 것은 `scripts/uninstall_pro360.ps1` 로 지운다.
- AFDA 프로젝트(github.com/jinw00ch01/Dacon_AFDA_Challenge)에서 가져온 교훈:
  - Store 앱(Claude 데스크톱) 터미널은 `%LOCALAPPDATA%` 쓰기를 앱 샌드박스로 돌린다 (이 Ultra 에서도 확인). 런타임 상태는 `C:\Dacon\WM_Runtime` 에 두고, 오래 도는 프로세스는 작업 스케줄러로만 띄운다.
  - GPU 드라이버 리셋(TDR)으로 학습이 죽을 수 있다. 체크포인트를 자주 저장하고, 죽은 작업은 `lost` 로 표시해 새 작업으로 재개한다.
  - 검증 데이터가 학습에 섞이지 않게 데이터셋 단위 분할을 고정한다 (`configs/splits/`).
- 무인 `claude -p` 루프(AFDA 방식)는 쓰지 않는다. Claude Code 권한 검사가 스스로 권한을 허용하는 에이전트 생성을 막았고, 에이전트는 사람이 보는 프로젝트 스레드로만 돈다.
