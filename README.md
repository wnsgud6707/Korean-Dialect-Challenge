# Korean Dialect Challenge

아주대학교 음성인식 과제 · 박준형(202020733)

한국어 음성으로 **경상도·전라도·충청도·강원도·제주도**를 분류하고, 방언을 표준어 텍스트로 변환합니다. 지역 분류는 Macro F1, 표준어 변환은 CER로 각각 평가합니다.

- [대회 소개 HTML](docs/index.html) — 다운로드한 뒤 브라우저에서 열 수 있습니다.
- [참가자 상세 가이드](docs/participant_v2_ko.md)
- [평가 기준 및 채점 방법](docs/leaderboard_v2_ko.md)
- [Baseline 결과](reports/baseline_recording_v2_2026-10-25.md)

## 1. 실행 환경

Python **3.12**를 권장합니다. `git`, `curl`이 필요합니다. 아래 Bash 명령은 Linux/WSL 기준이며 Windows PowerShell 명령도 함께 제공합니다. 로컬 검증은 Windows, Python 3.12.14, RTX 4070 12GB에서 진행했습니다. Linux/WSL의 전체 실행은 아직 검증하지 않았습니다.

```bash
git clone https://github.com/wnsgud6707/Korean-Dialect-Challenge.git
cd Korean-Dialect-Challenge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pilot.txt
```

Windows PowerShell:

```powershell
git clone https://github.com/wnsgud6707/Korean-Dialect-Challenge.git
cd Korean-Dialect-Challenge
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-pilot.txt
```

가상환경 활성화가 제한되면 이후 `python` 대신 `.\.venv\Scripts\python.exe`를 사용해도 됩니다. Bash에서는 `bash shell/setup.sh`로 환경 생성과 전처리 의존성 설치를 실행할 수 있습니다. 설치 후 가상환경을 활성화하세요.

### GPU 학습 환경

실측 환경은 PyTorch 2.6.0+cu124, transformers 4.49.0입니다. NVIDIA GPU에서 같은 버전을 사용하려면 아래 순서로 설치합니다. 드라이버와 CUDA wheel의 호환성은 본인 장비에서 확인하세요.

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-baseline.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CPU에서는 `python -m pip install -r requirements-baseline.txt`로 설치할 수 있지만 학습·추론 시간이 훨씬 길어집니다. `requirements-baseline-lock.txt`는 실측 환경의 패키지 목록입니다.

## 2. AI-Hub 이용 신청 및 API 키

[AI-Hub](https://www.aihub.or.kr/)에서 아래 다섯 데이터셋을 각각 신청하고 승인받아야 합니다. 승인 후 본인의 Shell API 키를 발급받습니다. **저장소에는 API 키가 없으며 참가자가 각자의 키를 사용합니다.**

| 지역 | datasetkey | 원천 filekey | 라벨 filekey |
|---|---:|---|---:|
| 강원도 | 118 | 572593 | 572589 |
| 경상도 | 119 | 572703 | 572701 |
| 전라도 | 120 | 572657 | 572654 |
| 제주도 | 121 | 572699, 572695 | 572694 |
| 충청도 | 122 | 572613 | 572612 |

키는 환경변수 `AIHUB_API_KEY`로 읽습니다. 화면과 Shell 기록에 키를 남기지 않으려면 아래처럼 입력합니다. `.env` 파일을 자동으로 읽는 방식은 아닙니다.

Bash:

```bash
read -r -s -p 'AI-Hub API key: ' AIHUB_API_KEY
printf '\n'
export AIHUB_API_KEY
```

PowerShell:

```powershell
$aihubSecret = Read-Host 'AI-Hub API key' -AsSecureString
$env:AIHUB_API_KEY = [System.Net.NetworkCredential]::new('', $aihubSecret).Password
Remove-Variable aihubSecret
```

키를 README, 코드, 커밋, Issue에 붙여 넣지 마세요. 다운로드 코드는 API 키를 명령 인자나 로그에 출력하지 않습니다.

## 3. 다운로드

```bash
python scripts/participant_v2.py plan
python scripts/participant_v2.py download
```

Bash에서는 `bash shell/download.sh`로 실행할 수 있습니다. 키가 설정되지 않았으면 숨김 입력으로 받습니다.

- 원천 ZIP 6개와 라벨 ZIP 5개, **총 11개**만 받습니다.
- 전송량 약 **66.88GB**. tar와 ZIP을 둘 다 보관하면 약 **133.77GB**이며 추출 음성·모델·환경을 위한 공간이 더 필요합니다.
- 저장 위치는 `data/downloads/`입니다. 이미 완성된 파일은 크기와 SHA256을 확인해 재사용합니다.
- 다운로드 중 끊긴 개별 파일은 처음부터 다시 받습니다. 다른 파일까지 다시 받지는 않습니다.
- 압축파일을 자동으로 삭제하지 않습니다. 데이터셋 전체를 받는 옵션은 제공하지 않습니다.

## 4. 전처리와 데이터 복원

```bash
python scripts/participant_v2.py extract
python scripts/participant_v2.py verify
python scripts/participant_v2.py export-text
```

Bash에서는 `bash shell/preprocess.sh`로 위 세 단계를 실행합니다.

원본 녹음에서 공개 목록에 지정된 발화 구간을 추출해 **16kHz / mono / PCM16 / 3~10초** WAV로 저장합니다. 임의로 다시 샘플링하지 않으므로 모든 참가자가 같은 발화를 사용합니다. SHA256이 다르면 검증이 실패합니다.

| 구분 | 발화 수 | 제공 내용 |
|---|---:|---|
| Train | 21,222 | 음성 복원 목록, 지역 라벨, 방언·표준어 문장 복원 |
| Validation | 3,120 | 음성 복원 목록, 지역 라벨, 방언·표준어 문장 복원 |
| Public Test | 3,043 | 음성 복원 목록, 지역 라벨; 표준어 정답은 운영자 평가 |
| Hidden Test | 3,058 | 운영자가 별도 보관; 선정 목록·정답 미제공 |

전체는 30,443개, 43.42시간, 약 5.00GB입니다. 참가자에게 복원되는 음성은 **27,385개**입니다. 결과는 `data/participant_recording_v2/`에 생성됩니다. Train/Validation 문장은 `text/`에 저장됩니다.

같은 녹음에서 추출한 발화와 동일한 음성 파일은 서로 다른 split에 들어가지 않습니다. 다만 다른 녹음에 같은 화자가 참여했는지는 확인할 수 없습니다. 지역별 용량을 약 1GB로 맞췄으며 발화 수·연령·성별은 동일하지 않습니다. 청취 검수는 진행하지 못했습니다.

AI-Hub 원본 음성과 전사 원문은 재배포하지 않습니다. 저장소의 JSONL에는 복원에 필요한 경로·시간·체크섬만 있고 전사 원문은 없습니다.

## 5. 방언 분류 baseline

Wav2Vec-BERT 2.0 Encoder를 고정하고 Classification Head를 학습합니다. Seed 42, 최대 10 epochs이며 Validation Macro F1로 체크포인트를 선택합니다.

```bash
python scripts/baseline.py preflight --config configs/baseline_participant_v2.json
python scripts/baseline.py train --config configs/baseline_participant_v2.json
python scripts/baseline.py evaluate --config configs/baseline_participant_v2.json --split public_test
```

각각 `bash shell/train.sh`, `bash shell/evaluate.sh`로도 실행할 수 있습니다. 분류 모델은 최초 실행 시 Hugging Face에서 고정 revision을 받습니다. 학습 결과는 `artifacts/baseline_participant_v2/best`에 저장합니다. RTX 4070 실측 학습 시간은 약 28분이며 초기 다운로드와 파일 검증은 제외한 값입니다.

## 6. 표준어 변환 baseline

Whisper large-v3-turbo로 전사하고 Train에서 학습한 어절 치환 규칙을 적용합니다. Whisper를 fine-tuning하거나 TTS를 수행하지 않습니다.

```bash
python experiments/standardization_v1/download_models.py asr_turbo
python experiments/recording_v2/standardize.py train --participant
python experiments/recording_v2/standardize.py asr --model asr_turbo --participant
```

`--participant`를 반드시 붙입니다. `standardization_v1`은 공통 모듈 경로이며 사용하는 데이터는 recording-v2입니다. 모델은 `data/standardization_models/asr_turbo`, 변환 규칙과 Validation 결과는 `data/standardization_participant_recording_v2`에 저장됩니다. 기존 규칙 출력은 덮어쓰지 않습니다.

## 7. 제출과 평가

운영자가 제공한 입력 JSONL(`sample_id`, `path`)을 읽어 아래 명령으로 두 작업의 예측을 생성합니다. `evaluation/inputs.jsonl`은 예시 경로이며 실제 제공 경로로 바꿉니다.

```bash
python experiments/recording_v2/predict_joint.py --inputs evaluation/inputs.jsonl --output predictions.jsonl --classifier artifacts/baseline_participant_v2/best --asr data/standardization_models/asr_turbo --editor data/standardization_participant_recording_v2
```

출력 형식(설명용 예시):

```json
{"sample_id":"example_0001","prediction":"gyeongsang","standard_text":"오늘 날씨가 좋습니다."}
```

- 지역 분류: **Macro F1**, Accuracy, 지역별 F1, confusion matrix.
- 표준어 변환: **CER**, WER, 지역별 CER. `1-CER`는 의미 정확도가 아닙니다.
- 속도: mean/p50/p95 latency, RTF. 완성된 WAV 파일을 읽어 두 예측을 완료할 때까지 측정하며 모델 로드·녹음·업로드 시간은 제외합니다.
- 같은 장비에서 속도를 비교하며, 분류와 변환 점수를 합산하지 않습니다.

[채점 코드](scripts/challenge_v2.py)와 [분류 채점](scripts/challenge_evaluation.py), [CER/WER 구현](experiments/standardization_v1/text_metrics.py)을 공개합니다. 운영자는 별도 보관한 정답과 평가 입력을 사용합니다. **채점 코드는 공개하지만 실제 비공개 평가 자료는 포함하지 않습니다.** 참가자는 Public Test 분류 평가와 Validation 표준어 평가를 로컬에서 실행할 수 있습니다.

```bash
# 운영자 전용: 별도로 보관한 평가 폴더가 있어야 실행됩니다.
python scripts/challenge_v2.py score --evaluation /path/to/organizer-evaluation --predictions predictions.jsonl --output metrics.json
```

기본 트랙은 지정 Train만 학습에 사용합니다. 원본 ZIP 전체나 외부 데이터로 추가 학습한 경우 별도로 표시합니다. 같은 원천 ZIP에 평가 음성이 존재할 수 있으므로 공식 목록 외 자료를 임의로 학습에 넣지 않습니다. 모델 설정은 Validation에서 선택하고 Test 결과로 튜닝하지 않습니다. 실제 접수 방법과 마감은 추후 안내합니다.

## 8. Baseline 성능

| 데이터 | Macro F1 ↑ | Accuracy ↑ | CER ↓ | WER ↓ |
|---|---:|---:|---:|---:|
| Validation | 90.03% | 90.13% | 19.95% | 35.26% |
| Public Test | 87.91% | 88.07% | 19.64% | 35.77% |
| Hidden Test | 87.48% | 87.57% | 17.87% | 33.53% |

Hidden Test 평균 처리 시간 0.174초, p95 0.236초, RTF 0.0340입니다. RTX 4070 12GB / batch 1 기준이며 실시간 스트리밍 지연을 뜻하지 않습니다. 공개 음성 전체의 재추출 및 체크섬 검증은 완료했습니다. 원본 ZIP 11개를 깨끗한 환경에서 전부 새로 다운로드한 검증은 아닙니다.

## 참고

- [HERIUN/korean_dialect_classification](https://github.com/HERIUN/korean_dialect_classification): AI-Hub 선택 다운로드와 방언 분류 구조 참고
- [AI-Hub](https://www.aihub.or.kr/): 데이터 이용 신청 및 원본 데이터
- [Wav2Vec-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0)
- [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- [교과목 과제 요구사항](https://github.com/LSH-Ajou/Speech-Recognition-Challenge-2026)

데이터와 사전학습 모델의 이용 조건은 각 제공처의 조건을 따릅니다. API 키, 원본 데이터, 생성된 전사, 모델 가중치, 비공개 평가 자료는 `.gitignore`로 제외합니다.
