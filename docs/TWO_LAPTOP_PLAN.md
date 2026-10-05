# 멀티 에이전트 운영 (Ultra 단독 기본, Pro 360 선택)

이전 AFDA 프로젝트(github.com/jinw00ch01/Dacon_AFDA_Challenge)의 교환·작업 실행 코드를 이 레포의 `wm_ops/` 로 옮기고, 문제였던 부분을 고쳤다.

## 0. Ultra 단독 운영 (기본, 2026-10-05 결정)

사용자가 Ultra 한 대로도 된다고 해서 기본을 Ultra 단독으로 바꿨다. 무거운 학습은 클라우드 GPU 가 맡고, Pro 360 은 GPU 가 없고 CPU 도 Ultra 보다 약해서 더하는 것이 적다.

```
     이 프로젝트 (claude.ai Projects) — 코디네이터 Claude 가 스레드 배정
            │                                  │
   ┌────────┴─────────┐   스레드 메시지   ┌────────┴──────────┐
   │ 리드 스레드       │◄──────────────►│ 검증·데이터 스레드  │   둘 다 Ultra 의 Claude 앱
   │ worktree A        │                 │ worktree B         │   (스레드마다 Worktree 옵션)
   └────────┬─────────┘                 └────────┬──────────┘
            │  open 정션 → C:\Dacon\RobotWorldModel_ActionVideo\open (원본 데이터)
            │  큰 산출물 → C:\Dacon\WM_Shared\
            └── wm_ops job start ──► WM-Exchange-ultra5060 (작업 스케줄러) ── GPU 작업 한 번에 하나
                                     └─► (선택) 클라우드 GPU
```

- 에이전트 = 이 프로젝트의 스레드 세션 두 개. 역할: 리드(`.claude/roles/ultra5060.md`), 검증·데이터(`.claude/roles/verifier.md`).
- 충돌 방지: 스레드마다 자기 worktree, 데이터는 `scripts\link_data.ps1` 정션(읽기 전용), GPU 는 대기열 + `C:\Dacon\WM_Runtime\gpu.lock`.
- 긴 작업은 `wm_ops job start` 로 Claude 앱 밖에서 돌고, 요청한 worktree 의 커밋된 코드로 실행된다.
- Pro 360 을 나중에 붙이려면 아래 3절의 붙여넣기 한 줄을 쓴다. 참여 코드는 Ultra 에서 `wm_ops pair-token new` 로 다시 만들 수 있다.

## 1. 두 대 구조 (Pro 360 을 붙일 때)

```
              이 프로젝트 (claude.ai Projects)
        코디네이터 Claude  ── 스레드 배정·메시지 ──┐
          │                                         │
   ┌──────┴─────────────┐                ┌──────────┴─────────┐
   │ Ultra 운영 스레드   │  스레드 간 메시지 │ Pro 운영 스레드     │
   │ (Claude 앱, 이 PC)  │◄───────────────►│ (Claude 앱, Pro PC) │
   └──────┬─────────────┘                └──────────┬─────────┘
          │ wm_ops publish / job start               │
   ┌──────┴─────────────┐   Syncthing     ┌──────────┴─────────┐
   │ WM-Exchange-ultra  │◄──────────────►│ WM-Exchange-pro    │  (작업 스케줄러, Claude 아님)
   │ Syncthing + tick   │  패킷·데이터     │ Syncthing + tick   │
   └──────┬─────────────┘                └──────────┬─────────┘
          └──────────── GitHub main (코드·문서·결정) ─┘
                     │
            (선택) 클라우드 GPU — Ultra 가 API 로 조작
```

- **에이전트 = 이 프로젝트의 Claude 세션.** 노트북마다 하나의 스레드 세션이 Claude 앱에서 돈다. 코디네이터가 일을 나누고, 두 세션은 스레드 메시지로 조율한다. 사람은 프로젝트 화면에서 모두 보고 끼어들 수 있다.
- **교환 서비스 = 작업 스케줄러 작업 `WM-Exchange-<role>`.** 로그온 시 시작, 죽으면 1분 뒤 재시작. 15초마다 `python -m wm_ops tick`:
  1. 상대가 보낸 패킷을 SHA-256 검증 후 `work/packets/inbox/` 로 가져오고 ack 를 보낸다.
  2. 세션이 대기열에 넣은 긴 작업(`wm_ops job start`)을 Claude 앱 밖에서 띄우고 감시한다.
  3. 깨끗한 `main` 이면 5분마다 fast-forward 한다.
  4. 상대 장치를 페어링한다: `configs/nodes.json` 에 올라온 장치 ID, 또는 일회용 참여 코드로 이름을 알린 대기 장치.
  5. 상대가 읽을 수 있는 상태 파일(`status/<role>.json`)을 쓴다.
- **Syncthing 폴더** (전용 인스턴스, 관리 API 는 127.0.0.1 만):
  - `C:\Dacon\WM_Exchange\ultra_to_pro` (Ultra 송신 전용), `pro_to_ultra` (Pro 송신 전용): 패킷.
  - `open\` (Ultra → Pro): 대회 데이터 8.65GB. Pro 디스크가 작으면 학습 영상 8.34GB 를 건너뛴다.
- **Git** 은 코드와 결정, 장치 ID 등록의 신뢰 경로다.

## 2. 역할

| | Ultra 운영 (리드) | Pro 운영 (데이터·검증) |
|---|---|---|
| 기기 | Core Ultra 7, RAM 32GB, RTX 5060 Laptop 8GB, 여유 430GB | i7-1360P, RAM 16GB, Iris Xe (CUDA 없음) |
| 일 | 모델 코드·통합, GPU 작업, 제출 후보 mp4, 클라우드 GPU 조작, `main` 통합 | 데이터 인덱스·분할, CPU 재계산 검증, 영상 QA, 리뷰 |
| 금지 | 제출킷 모델을 학습·선택에 사용 | GPU 작업, `open/` 쓰기 |
| git | `main` 푸시 | `pro/<주제>` 브랜치 |

사람: DACON 업로드, 클라우드 계정·결제·API 키, 백본 라이선스 최종 확인.

## 3. 설치 상태

### Ultra (완료)

```
scripts\setup_env.ps1 -Role ultra5060 -Kit     # .venv-ultra5060 (torch 2.8.0+cu128), .venv-kit (제출킷 고정 버전)
scripts\setup_node.ps1 -Role ultra5060         # Syncthing, configs/local-node.json, 작업 WM-Exchange-ultra5060, nodes.json
```

### Pro 360 설치 (붙여넣기 한 번)

Remote Control 이 Pro 360 에 닿지 않아, Pro 에서 PowerShell 창에 명령 한 줄을 붙여넣는 방식으로 바꿨다. 관리자 권한도 GitHub 로그인도 필요 없다.

1. Ultra 에서 일회용 참여 코드를 만든다: `.venv-ultra5060\Scripts\python.exe -m wm_ops pair-token new` (기본 48시간 유효, 한 번 쓰면 사라짐). 출력의 `peer_command` 가 붙여넣을 명령이다.
2. Pro 의 Windows PowerShell 에 그 명령을 붙여넣는다. `scripts/bootstrap_pro360.ps1` 이 차례로 한다:
   - git 이 없으면 MinGit 2.56.0, Python 3.12 가 없으면 python.org 3.12.10 을 사용자 폴더에 설치 (둘 다 SHA-256 고정 검증, PATH 변경 없음)
   - 공개 레포를 `C:\Dacon\RobotWorldModel_ActionVideo` 로 받기 (이미 있으면 갱신)
   - `setup_env.ps1 -Role pro360` (CPU torch), `setup_node.ps1 -Role pro360` (Syncthing, 작업 `WM-Exchange-pro360`). 여유 디스크 25GB 미만이면 학습 영상 동기화를 건너뛴다.
   - Pro 는 Ultra 장치 ID 를 `configs/nodes.json` 에서 읽고, 자기 이름을 `WM-pro360-<참여 코드>` 로 알린다. Ultra 교환 서비스는 그 이름의 대기 장치 하나만 수락하고 코드를 지운다. 연결되면 Pro 는 이름을 `WM-pro360` 으로 되돌린다.
3. 확인: 양쪽에서 `python -m wm_ops status` 의 `peer_connected` 와 `peer_heartbeat`.

이 페어링 경로는 임시 Syncthing 두 개로 시험했다: 수락 9초, 패킷 전달 12초, ack 회신 12초.

## 4. 무인 `claude -p` 루프 (AFDA 방식)는 설치하지 않았다

AFDA 는 노트북마다 `claude -p` 를 주기적으로 띄우고, 훅이 명령을 자동 허용하는 무인 루프를 썼다. 이번에 이 세션에서 같은 방식을 시험하려 하자 Claude Code 권한 검사가 "스스로 권한을 허용하는 에이전트 생성"으로 막았다. 그래서 위 구조에서는 Claude 가 사람의 권한 설정 안에서 도는 프로젝트 세션으로만 일한다. 무인 루프가 꼭 필요하면 사람이 직접 결정하고 켜야 한다.

## 5. AFDA 에서 배운 것 → 이번 대책

| AFDA 에서 일어난 일 | 이번 대책 |
|---|---|
| Store 앱 터미널이 `%LOCALAPPDATA%` 쓰기를 앱 샌드박스로 돌려 서비스가 상태를 못 봄 (이 Ultra 에서도 재현 확인) | 런타임 상태를 `C:\Dacon\WM_Runtime` 에 둠, 서비스는 작업 스케줄러로만 실행, HKCU Run 키 미사용 |
| 교환 프로세스가 0xC000013A 로 죽었는데 3시간 방치 | 작업 스케줄러 재시작(1분 간격) + Syncthing 프로세스 재사용·재기동 감시 |
| GPU 드라이버 리셋(TDR)으로 학습·교환·앱이 함께 죽음 | GPU 작업은 Claude 앱·교환 서비스와 분리된 프로세스, 체크포인트 주기 저장, 죽으면 `lost` 로 표시해 새 작업으로 재시작 |
| Syncthing 페어링 실패·충돌 사본 | 버전 고정(v2.1.5, SHA-256), 송신/수신 전용 폴더, 수신 폴더 버저닝 제거, 장치 ID 는 git 또는 일회용 참여 코드로 교환 |
| 검증 데이터가 학습에 섞임 | 데이터셋 단위 분할 manifest 를 Pro 가 만들고 해시로 고정 |
| 두 기기가 한 계정 사용량을 공유 | 무인 루프 없이 사람이 보는 세션만 사용 |
