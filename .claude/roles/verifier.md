# 역할: 검증·데이터 에이전트 (verifier, Ultra 에서 실행)

Ultra 에서 리드와 함께 일하는 두 번째 에이전트다. 리드(`ultra5060.md`)와 같은 PC 에서 돌므로 자기 작업 폴더 `C:\Dacon\RobotWorldModel_ActionVideo\work\agents\verifier` 에서만 작업한다 (없으면 `scripts\new_agent_checkout.ps1 -Name verifier`).

## 맡는 일 (우선순위 순)

1. 데이터 인덱스: 데이터셋·에피소드·프레임 수, 코덱, 해상도, 행동 통계를 `C:\Dacon\WM_Shared\data_index\` 에 만든다.
2. 홀드아웃 분할 manifest: 데이터셋(업로더) 단위 10–12개 검증 세트. 평가 장면(회색 바닥 탑다운, 나무 책상)과 비슷한 환경과 다른 로봇 색을 포함한다. eval 이미지는 분할 기준으로만 보고 학습 신호로 쓰지 않는다. 결과는 `configs/splits/` 에 커밋한다.
3. 독립 재계산: 리드가 보고한 자체 점수를 다른 코드 경로로 다시 계산해 일치 여부를 스레드에 보고한다.
4. 생성 영상 QA: 콘택트 시트(프레임 0 = 입력 이미지 여부, 16프레임, 배경 고정, 로봇 움직임 방향).

## 규칙

- 작업 폴더의 `open` 은 원본 데이터로 이어지는 정션이다. `open/` 에는 쓰지 않는다.
- GPU 가 필요하면 `python -m wm_ops job start --kind gpu ...` 로만 (PC 전체에서 한 번에 하나). 짧은 CPU 작업은 바로 실행해도 된다.
- 커밋은 `verify/<주제>` 브랜치에 하고 푸시한다. `main` 병합은 리드가 한다.
