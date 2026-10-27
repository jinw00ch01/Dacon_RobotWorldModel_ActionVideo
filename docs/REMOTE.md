# 다른 노트북에서 할 일 (원격 작업 안내)

Ultra(이 PC)는 켜 두고 AC 전원에 연결해 둔다. Claude 는 Ultra 에서 계속 일하고, 진우님은 다른 노트북의 브라우저와 Claude 앱만 쓰면 된다.

## 1. Colab v5 학습 (지금 할 일)

1. 노트북 열기: https://colab.research.google.com/github/jinw00ch01/Dacon_RobotWorldModel_ActionVideo/blob/main/colab/train_cosmos_ac_v5.ipynb
   (링크가 안 열리면 스레드에 첨부한 `train_cosmos_ac_v5.ipynb` 를 Colab 의 "파일 > 노트북 업로드" 로 연다.)
2. 런타임 > 런타임 유형 변경 > H100 (없으면 A100). H100 은 약 5시간, A100 40GB 는 약 11시간.
3. 런타임 > 모두 실행. Google Drive 접근 허용 창이 뜨면 허용한다.
4. "잠재 캐시가 n/116 개" 또는 "업로드가 끝나지 않았습니다" 로 멈추면 Drive 업로드가 아직 덜 된 것이다.
   drive.google.com 에서 `dacon_wm/latents_240x320_extra` 에 파일 117개가 모두 보이면 다시 모두 실행한다.
5. 끊기면 다시 모두 실행한다. 잠재 IDM 은 건너뛰고 마지막 체크포인트(500 스텝마다 저장)부터 이어진다.
6. 진행 확인: Drive `dacon_wm/cosmos_ac/runs/v5/train_log.jsonl` (50 스텝마다 한 줄). 16,000 에서 32,000 스텝까지 간다.

## 2. 학습이 끝난 뒤

진우님이 할 일은 없다. Ultra 가 Drive 를 보고 있다가 마지막 4개 체크포인트 평균, 판정(STATUS 23), 제출 8 CSV 생성을 한다 (Ultra GPU 약 3시간).
판정을 통과하면 이 스레드에 CSV 를 첨부한다.

## 3. 제출 파일 받기와 업로드

- 스레드에 첨부된 CSV 를 다른 노트북에서 내려받아 DACON 에 업로드한다.
- 점수(Public, Private)를 스레드에 알려 주면 STATUS 에 기록하고 다음 단계를 정한다.
- 지금까지 최고: 제출 5 (Public 0.2263 / Private 0.2141).

## 4. 주의

- Ultra 가 재부팅되면 GPU 작업은 로그인 때 자동으로 이어지지만, 이 스레드는 Ultra 에서 Claude 앱을 열어야 다시 연결된다. 원격으로는 재부팅하지 않는다.
- Drive 의 `dacon_wm` 폴더는 지우지 않는다 (학습 입력과 v5 체크포인트가 있다).
