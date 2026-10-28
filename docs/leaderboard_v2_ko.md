# recording-v2 Leaderboard 운영 설계

분류와 표준어 변환의 순위를 분리한다. 하나의 임의 합성 점수로 모델 크기·속도·정확도를 섞지 않는다.

| 트랙 | 순위 기준 | 함께 표시 |
|---|---|---|
| 5개 지역 방언 식별 | Macro F1 높은 순 | Accuracy, 지역별 F1, confusion matrix |
| 음성 → 표준어 텍스트 | CER 낮은 순 | WER, 지역별 CER, 의미 보존 예시 |
| 처리 속도 참고 | 동일 환경에서 측정 | 전체 mean/p50/p95 latency, RTF, 장비, precision, batch size |

공식 baseline은 두 기능을 함께 실행하는 reference predictor를 제공한다. 참가자는 개선한 코드로 정답 없는 `inputs.jsonl`을 읽고 `sample_id`, `prediction`, `standard_text`만 가진 JSONL을 출력한다. 평가 코드는 모델을 실행하지 않고 제출 예측을 검증·채점한다. 누락·추가·중복 ID, 잘못된 지역 라벨, 정해진 형식 외 필드는 거부한다.

CER/WER는 NFC 정규화, 문장부호를 공백으로 변환, 연속 공백 정리를 적용한다. CER는 공백을 제외한 문자, WER는 공백 기준 어절을 대상으로 전체 오류 수 / 전체 정답 길이를 계산한다. 1-CER를 번역 정확도라고 부르지 않는다. CER는 의미가 같은 다른 표현도 오류로 계산하며, 비식별 라벨과 숫자 표기 차이의 영향도 있다. 문장 완전 일치율은 개발 진단에만 사용하며 주 순위 기준으로 삼지 않는다.

시간은 batch 1, 1회 워밍업 후 WAV 읽기 → 특징 추출 → 지역 분류 → 한국어 전사 → 표준어 치환까지 측정한다. 모델 로딩과 출력 JSON 쓰기는 제외하고 별도 설명한다. GPU는 측정 구간 전후 synchronize한다. RTF는 전체 처리 시간 / 전체 입력 음성 길이다. batch 추론 처리량을 단일 요청 latency와 혼용하지 않는다. 공식 시간은 운영자가 동일 장비에서 다시 측정하며 참가자 제공 timing 파일만으로 순위를 확정하지 않는다.

완성된 WAV 파일이 로컬에 있는 조건이며 실제 발화의 녹음 시간과 네트워크 업로드 시간은 제외한다. 이 값은 스트리밍 음성인식의 첫 응답 지연을 의미하지 않는다.

## 운영자 평가

```sh
python scripts/challenge_v2.py prepare --split public_test --output data/evaluation_joint_recording_v2_public
python scripts/challenge_v2.py prepare --split hidden_test --organizer --output data/evaluation_joint_recording_v2_hidden
python scripts/challenge_v2.py score --evaluation data/evaluation_joint_recording_v2_public --predictions predictions.jsonl --output metrics.json
```

prepare는 새 출력 폴더에만 실행한다. 음성 파일명을 무작위 순서의 중립 ID로 바꾸고 정답은 `private/`에 둔다. 평가 입력과 정답의 체크섬을 고정한다. 표준어 정답은 운영자 선정 목록의 동일 발화에서 가져오며 참가자에게 전달하지 않는다. 입력·정답이 한 상위 폴더에 있어도 평가 환경의 보안 격리가 자동으로 생기는 것은 아니다. 참가자 코드를 실행할 때는 입력만 있는 별도 환경을 사용한다.

고정 Train만 사용하는 폐쇄 데이터 트랙을 기본으로 한다. 원천 AI-Hub ZIP 전체 또는 외부 자료로 추가 학습하는 제출은 별도 표시한다. 사전학습 모델 사용은 허용하되 모델명·revision·학습 설정·추가 데이터 사용 여부를 기록한다. 동일한 원본 ZIP에 평가 음성이 존재할 수 있으므로 원천 자체가 비공개라고 주장하지 않는다.

홈페이지는 docs/index.html에 있습니다. 자동 접수 서버는 없으며 채점과 Leaderboard 갱신은 운영자가 진행합니다. 제출 방법과 마감은 추후 안내합니다. 운영자 prepare 명령은 이 저장소에 포함하지 않은 원본 선정 목록과 정답 자료가 있어야 실행됩니다.
