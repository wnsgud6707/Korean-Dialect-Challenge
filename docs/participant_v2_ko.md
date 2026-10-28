# recording-v2 참가자 재현 절차

recording-v2 참가자용 실행 절차입니다. recording-v1과 별도 버전입니다.

Python 3.12 환경을 권장한다(실측 3.12.14). 고정 NumPy 의존성 때문에 최소 Python 3.11이 필요하다. 음성 복원은 `pip install -r requirements-pilot.txt`, 모델 실행은 `pip install -r requirements-baseline.txt` 환경을 사용한다. GPU 학습에는 장비와 드라이버에 맞는 CUDA 지원 PyTorch가 필요하다. 현재 실측은 RTX 4070 12GB, torch 2.6.0+cu124, transformers 4.49.0 조건이며 CPU 실행 시간과 다르다. `python -c "import torch; print(torch.cuda.is_available())"`로 CUDA 사용 가능 여부를 확인한다.

1. AI-Hub 계정으로 datasetkey 118~122 각각 이용 신청·승인을 받는다.
2. 본인의 API 키를 로컬 환경변수 `AIHUB_API_KEY`에 설정한다. 키를 코드·채팅·GitHub에 올리지 않는다.
3. 배포된 공개 메타데이터와 같은 버전의 코드 및 `requirements-pilot.txt` 의존성을 준비한다.
4. 아래 순서로 지정 ZIP 다운로드, 음성 복원, 검증을 실행한다. Python 명령은 Windows/Linux Shell 모두에서 실행할 수 있으며 본인의 Python 환경 이름을 사용한다.

```sh
python scripts/participant_v2.py plan
python scripts/participant_v2.py download
python scripts/participant_v2.py extract
python scripts/participant_v2.py verify
python scripts/participant_v2.py export-text
```

원천 ZIP 6개와 라벨 ZIP 5개, 총 11개만 받는다. 새로 받는 전송량은 약 66.88GB다. tar와 ZIP을 모두 보관하면 원천 파일만 약 133.77GB이므로 음성 출력·모델·환경까지 고려해 디스크 여유를 확보한다. 기존 파일이 있으면 정확한 크기·SHA256을 확인하고 재사용한다. 자동 삭제는 하지 않는다.

복원 결과는 `data/participant_recording_v2/`의 공개 27,385개 음성이다. Train 21,222 / Validation 3,120 / 공개 Test 3,043개로 고정된다. 숨겨진 3,058개를 고르는 목록과 정답은 제공하지 않는다. 같은 AI-Hub 원천 ZIP 안에 평가용 원본이 들어 있을 수 있으나 공식 공개 목록 외 자료를 학습에 사용하는 것은 별도 트랙으로 구분해야 한다. 폐쇄 데이터 트랙은 지정 Train으로 학습하고 Validation으로 설정을 선택한다.

`export-text`는 내려받은 라벨에서 Train/Validation의 방언형·표준형 정답을 로컬 복원한다. 이 전사 파일과 원본 음성을 GitHub에 재배포하지 않는다. 공개 메타데이터 자체에는 전사 원문을 담지 않는다.

```sh
python scripts/baseline.py preflight --config configs/baseline_participant_v2.json
python scripts/baseline.py train --config configs/baseline_participant_v2.json
python scripts/baseline.py evaluate --config configs/baseline_participant_v2.json --split public_test
```

학습에는 `requirements-baseline.txt`의 환경과 해당 모델이 추가로 필요하다. 위 baseline은 Encoder 고정 지역 분류다. 표준어 변환 baseline 및 처리 시간 평가는 별도 실험 코드와 결과를 확인한다. 30분의 학습 시간을 보장하지 않는다. 공개 평가 결과로 숨겨진 평가 점수를 추정하거나 다른 데이터 버전의 점수와 직접 비교하지 않는다.

## 표준어 변환 baseline

공식 baseline의 전사 모델은 Validation CER로 선택한 `asr_turbo` (Whisper large-v3-turbo)다. 비교용 Whisper-small은 `asr` 인자로 실행할 수 있다. 모델 revision은 다운로드 코드에 고정되어 있다. 전사 모델 자체는 추가 학습하지 않으며, 지정 Train의 문장 쌍에서 변환 규칙을 학습한다. Validation의 CER로 규칙 설정을 고른다.

```sh
python experiments/standardization_v1/download_models.py asr_turbo
python experiments/recording_v2/standardize.py train --participant
python experiments/recording_v2/standardize.py asr --model asr_turbo --participant
```

중간 결과는 `data/standardization_participant_recording_v2/`에 저장한다. train 명령은 기존 결과를 덮어쓰지 않는다. 다른 모델·규칙·데이터로 실험할 때는 새 버전을 만들어 캐시를 혼합하지 않는다. 현재 참가자 경로로 학습한 규칙 및 텍스트 validation 결과는 운영자 경로 결과와 바이트 단위로 일치함을 확인했다.

## 두 기능을 함께 제출하는 인터페이스

운영자 제공 입력은 `sample_id`와 음성 상대경로 `path`만 포함한다. 예측 JSONL의 각 줄은 `sample_id`, `prediction`(지역 영문 키), `standard_text`만 포함해야 한다. 누락·추가·중복 ID는 채점 오류다. 지역 분류는 Macro F1, 표준어 출력은 CER로 각각 순위를 정하고 합성 점수로 섞지 않는다. WER와 지역별 지표는 보조로 제공한다.

```sh
python experiments/recording_v2/predict_joint.py --inputs evaluation/inputs.jsonl --output predictions.jsonl --classifier artifacts/baseline_participant_v2/best --asr data/standardization_models/asr_turbo --editor data/standardization_participant_recording_v2
```

`predictions.timing.json`에는 batch 1의 평균·중앙·p95 처리 시간과 전체 RTF가 기록된다. WAV 읽기, 특징 추출, 분류, 전사, 표준어 치환을 포함하며 모델 로드·1회 워밍업·JSON 파일 쓰기는 제외한다. 공식 속도 비교는 운영자가 같은 장비와 환경에서 재측정해야 하며 참가자 자체 보고를 그대로 순위에 쓰지 않는다. 채점 스크립트는 공개하며 실제 private 정답과 평가 선정 목록은 제공하지 않습니다.
