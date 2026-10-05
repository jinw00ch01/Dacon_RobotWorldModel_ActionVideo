# 클라우드 GPU 대여 확인 (2026-10-05 약 09:00 UTC 실시간 조회)

결론: **대회 기준 GPU(RTX PRO 6000 96GB)를 지금 바로 빌릴 수 있다.** 다만 계정 생성·선불 충전·API 키는 사람이 해야 한다.

## 실시간으로 확인한 것

| 제공자 | GPU | 가격 (1장, 시간당) | 재고 | 확인 경로 |
|---|---|---|---|---|
| RunPod Secure Cloud | RTX PRO 6000 Server 96GB | $2.09 | High | api.runpod.io/graphql (공개 조회) |
| RunPod Community | RTX PRO 6000 WS / Max-Q | $1.69 / $1.64 | Low | 같음 |
| Vast.ai (검증 호스트, 디스크 ≥300GB, 신뢰도 ≥0.98) | RTX PRO 6000 WS/S | $1.42–1.59 | 29개 | console.vast.ai/api/v0/bundles (공개 조회) |
| Vast.ai 일본 | RTX PRO 6000 WS | $1.62 (300GB 포함) | 1개 (offer 54008577) | 같음 |
| RunPod | H100 SXM / PCIe | $3.49 / $2.89 | Medium / Low | 같음 |
| Vast.ai | H100 | $2.00부터 (중앙값 약 $2.9) | 8개 검증 | 같음 |
| RunPod / Vast.ai | A100 80GB | $1.59 / $0.80–1.00 (디스크 300GB 이상은 $1.11부터) | High / 12개 | 같음 |
| 엘리스 클라우드 | A100 80GB / H100 | ₩2,500 / ₩5,500 (VAT 별도) | 미확인 | 가격 피드만, API 없음 |

- RunPod 의 RTX PRO 6000 재고는 미국·캐나다·유럽 데이터센터뿐이고 아시아에는 없다.
- Vast.ai 는 데이터 전송이 바이트 과금(대개 입력 $2.67/TB)이고, 인스턴스를 멈추면 GPU 를 다른 사람이 가져갈 수 있다.
- RunPod 은 전송 과금이 없고, 네트워크 볼륨(300GB 약 $21/월)에 데이터를 두면 포드를 지웠다 다시 만들어도 데이터가 남는다.

## 비용 추정 (₩1,344.6/USD)

| 선택 | 48 GPU시간 | 96 GPU시간 |
|---|---|---|
| RunPod RTX PRO 6000 $2.09 + 볼륨 | 약 $110 (₩148k) | 약 $211 (₩284k) |
| Vast.ai RTX PRO 6000 $1.42–1.62 | 약 $68–78 (₩92–105k) | 약 $136–155 (₩183–209k) |

## 추천

1. **RunPod Secure Cloud, RTX PRO 6000 1장 + 300GB 네트워크 볼륨.** 대회 기준 GPU 와 같고, 재고가 많고, `runpodctl`/API 로 에이전트가 포드를 만들고 지울 수 있다.
2. 비용을 줄이려면 Vast.ai 검증 호스트의 RTX PRO 6000. 학습 중 체크포인트를 계속 밖으로 복사해야 한다.

## 사람이 직접 해야 하는 일

1. RunPod (또는 Vast.ai) 가입, 약관 동의, 이메일 인증.
2. 본인 카드로 선불 충전 (RunPod 48시간분 약 $120). 자동 충전은 꺼 두면 지출 상한이 된다.
3. API 키 발급 (RunPod: Settings → API Keys). 채팅에 붙이지 말고 Ultra 의 사용자 환경 변수 `RUNPOD_API_KEY` 로 저장한다.
4. Hugging Face 로그인 → `nvidia/Cosmos-Predict2.5-2B` 라이선스 동의(자동 승인) → read 토큰을 환경 변수 `HF_TOKEN` 으로 저장.

그 다음 단계(포드·볼륨 생성, 데이터 업로드, 학습 시작·중지, 체크포인트 회수)는 에이전트가 API/CLI 로 한다.

## 확인하지 못한 것

- 실제 기동 성공 여부(계정이 있어야 확인 가능). 재고는 분 단위로 바뀐다.
- 최소 충전 금액(RunPod 은 선불카드만 $100 명시).
- Lambda, VESSL, TensorDock 등은 로그인이나 키가 있어야 재고를 볼 수 있어 확인하지 못했다.
