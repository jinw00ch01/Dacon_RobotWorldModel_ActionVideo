# 노트북 2대 멀티 에이전트 운영 계획

이전 AFDA 프로젝트(github.com/jinw00ch01/Dacon_AFDA_Challenge)의 `agent_bridge`, `exchange_bridge`, `docs/JOURNEY.md` 기록을 바탕으로, 잘 된 것은 그대로 가져오고 문제였던 부분을 고친다.
**아직 아무것도 설치·설정하지 않았다.** 아래 "설치 단계"는 사용자 확인 후 진행한다.

## 1. 역할

| | Ultra 운영 (리드) | Pro 운영 (데이터·검증) |
|---|---|---|
| 기기 | Galaxy Book6 Ultra, Core Ultra 7, RAM 32GB, RTX 5060 Laptop 8GB (드라이버 591.74), torch 2.12 cu128 확인 | Galaxy Book3 Pro 360, i7-1360P, RAM 16GB, Iris Xe(CUDA 없음), AFDA 당시 여유 디스크 약 24GB |
| 맡는 일 | 모델 코드·통합, GPU 작업(추론, 저해상도 LoRA 시험, 특징 추출), 제출 후보 mp4 생성, `main` 병합 | 데이터 인덱스·분할 manifest, AV1 디코딩 캐시(저해상도), 자체 채점기 CPU 재계산, 생성 영상 QA(콘택트 시트), 코드 리뷰 |
| 금지 | 제출킷 모델을 학습·선택에 사용 | 학습 작업, 대용량 원본 복제 |
| git | `main` 푸시 권한 | `pro/<주제>` 브랜치 → Ultra 가 리뷰 후 병합 |

클라우드 GPU(트랙 A)를 쓰기로 하면 **세 번째 실행 노드**로 붙인다. 에이전트는 두지 않고, Ultra 가 학습 작업을 원격으로 띄우고 결과(체크포인트, 로그)를 받아온다.

사람이 할 일: DACON 업로드(에이전트는 업로드하지 않음), 클라우드 비용 결정, 백본 라이선스 최종 확인, 하루 제출 횟수 배분.

## 2. 통신 구조

```
            GitHub (코드·문서·실험 기록, 단일 진실)
             ▲  main            ▲ pro/<topic>
             │                  │
   ┌─────────┴───────┐   ┌──────┴──────────┐
   │ Ultra 운영       │   │ Pro 운영         │
   │ agent loop      │◄─►│ agent loop      │   Syncthing 교환 폴더
   │ GPU job queue   │   │ CPU job queue   │   C:\Dacon\WM_Exchange\
   └────────┬────────┘   └─────────────────┘     ultra_to_pro\ (Ultra 송신 전용)
            │ (선택) 원격 학습                     pro_to_ultra\ (Pro 송신 전용)
            ▼
     클라우드 GPU 48–96GB
```

- **git = 코드와 결정.** 실험 설정·결과 요약·결정은 모두 커밋으로 남긴다.
- **Syncthing = 큰 산출물만.** 패킷(`spec`/`result`/`review`/`qa`/`request`/`ack`), 샘플 mp4, 특징 캐시, 소형 체크포인트. 원본 `open/` 데이터는 동기화하지 않고 각자 `open.zip` 을 한 번 풀어 쓴다(Pro 는 디스크가 부족하면 `open/data/eval` + 메타데이터 + parquet + 일부 영상만).
- 패킷 형식은 AFDA `afda.experiment.v1` 를 이름만 바꿔(`wm.experiment.v1`) 재사용: `<id>/{kind.json, files/, manifest.json, COMMITTED.json}`, SHA-256 검증 후 ack, 실행 파일 첨부 거부, 정정은 `supersedes`.
- 사람과의 대화는 이 프로젝트 스레드가 맡고, 두 노트북 운영 세션 사이 메시지는 cross-session 메시지로 주고받는다.

## 3. 에이전트 루프 (AFDA `agent_bridge` 이식)

- `python -m agent_bridge loop` 20초 틱, 깨울 이유(새 패킷, 작업 종료, 입력 변경, 요청된 `next_wake`, 90분 하트비트)가 있을 때만 `claude -p` 실행. 실패 시 2분→1시간 백오프.
- 10분 넘는 일은 detached job 으로. GPU 작업은 한 번에 하나, 커밋된 코드에서만.
- 예산: 사이클당·일일 상한, 하루 사이클 수 상한, 사용량 한도 도달 시 리셋 시각까지 대기 (`configs/agent_policy.json`).
- 매 실행을 `harness` 로 기록(설정, 코드 해시, RAM, 시간) → `runs/`.
- `.claude/hooks/guard.py` 허용·차단 규칙에 이번 대회 전용 규칙 추가: `submission_kit/` 수정 금지, 제출킷 모델·체크포인트를 학습/추론 코드에서 import 금지(Rule 4·7), `data/eval` 을 학습 데이터로 읽기 금지(Rule 3).

## 4. AFDA 에서 배운 것 → 이번에 바꾸는 점

| AFDA 에서 일어난 일 | 이번 대책 |
|---|---|
| 앱(MSIX) 안 터미널이 `%LOCALAPPDATA%` 를 샌드박스로 돌려 설정이 실제 루프에 안 닿음 | 설치 스크립트는 일반 PowerShell 에서 실행, 설치 후 실제 경로를 검증하는 `check.ps1` |
| 교환 프로세스가 0xC000013A 로 죽었는데 작업 스케줄러가 실패로 안 봐서 3시간 정지 | 처음부터 상호 감시(루프↔교환, 상태 파일 2분/10분 기준), `STOP` 파일로만 정지 |
| GPU 드라이버 리셋(TDR)으로 학습·교환·앱이 함께 죽음 | GPU 작업은 짧은 주기 체크포인트 + 자동 재개, 교환 서비스와 분리된 프로세스, 노트북 전원·발열 설정 확인 |
| Syncthing 페어링·충돌 사본 문제 | 버전 고정 + 송신 전용/수신 전용 폴더 + 상대 기기 ID 로 설정, 상태 파일은 기기별로 분리 |
| 검증 데이터가 학습에 섞임 | 데이터셋 단위 분할 manifest 를 Pro 가 만들고 Ultra 가 해시로 고정 |
| 실행 중 사이클이 사람의 정정을 덮어씀 | 사이클 종료 후 재깨우기 감시 유지 |
| 에이전트가 사람 의도보다 보수적으로 제출 계획 | 제출 계획은 사람이 고르는 후보 목록 형태로만 제안 |
| 일시정지 중 마감 경과 | 이번엔 마감 걱정은 없지만, 정지 상태가 길면 사람에게 알림 |
| 두 기기가 한 계정 사용량을 공유해 한도 도달 | 루프 일일 예산을 기기별로 나누고, Pro 는 저빈도(하트비트 위주) |

## 5. 설치 단계 (사용자 확인 후 진행)

1. 레포에 `agent_bridge/`, `exchange_bridge/`, `harness/`, `.claude/` 를 AFDA 에서 이식해 이름·경로·규칙을 이번 대회에 맞게 수정 (PR).
2. 두 노트북에 같은 Python 3.12 가상환경. Ultra 는 CUDA torch, Pro 는 CPU torch. 의존성: polars/pyarrow, opencv, av, timm, diffusers 등.
3. Pro 에 데이터 배치(`open.zip` 복사 또는 부분 복사), 디스크 여유 확인.
4. Syncthing 설치·페어링(교환 폴더 2개만).
5. 작업 스케줄러 등록: `WM-Agent-<role>`, `WM-Exchange-<role>` (로그온 시 시작, 1분 간격 재시작, 관리자 권한 불필요).
6. 첫 왕복 시험: Ultra 가 `spec` 패킷 → Pro 가 분할 manifest `result` 회신 → Ultra 가 ack.
7. S0 제출(첫 프레임 반복)로 전체 경로 확인.
