# LSeg Image Encoder ONNX Extraction

이 프로젝트는 LSeg 모델의 이미지 인코딩 경로(백본, 중간 feature 추출, projection layer)를 분리하여 ONNX 모델로 변환하는 코드입니다.  
원본 LSeg 구조와 가중치(ckpt)는 그대로 유지하며, 추론용으로 ONNX로 export할 수 있도록 구성되어 있습니다.

## Installation

### 1. Python Version
- Python 3.8 이상 권장

### 2. Dependencies

아래 명령어를 사용하여 필요한 라이브러리를 설치합니다:

```bash
pip install torch torchvision torchaudio
pip install timm
pip install git+https://github.com/openai/CLIP.git
pip install pytorch_lightning
```

필요에 따라 가상환경(venv, conda 등)을 사용하세요.

## Project Structure 


```bash
.
├── demo_e200.ckpt          # 모델 체크포인트 (대용량 파일, .gitignore에 포함됨)
├── extract.py              # ONNX 변환을 실행하는 스크립트
└── langSeg                 # LSeg 관련 모듈 폴더
    ├── __init__.py         # 빈 파일 (패키지 인식용)
    ├── lseg_blocks.py      # LSeg 블록 및 백본 관련 함수 (경량화된 버전)
    ├── lseg_module.py      # 모델 로드 및 래핑 모듈
    ├── lseg_net.py         # LSegNet 정의 (이미지 인코딩 경로만 포함)
    └── lseg_vit.py         # CLIP 비주얼 백본 및 관련 함수들
```

## Usage 

프로젝트 루트 디렉토리에서 ONNX 모델을 추출하려면:
 
1. **체크포인트 로드 및 ONNX 내보내기** `extract.py` 스크립트를 실행합니다.

```bash
python3 extract.py
```

이 스크립트는 아래 단계를 수행합니다:
 
  - `demo_e200.ckpt` 파일에서 가중치를 로드하여 모델을 복원합니다.
 
  - 모델의 이미지 인코딩 경로(백본 → 중간 feature 추출 → projection layer)를 그대로 사용해 `LSegImageEncoder` 모듈을 생성합니다.
 
  - 더미 입력 텐서 (1, 3, 480, 480)를 사용하여 ONNX 모델 (`lseg_image_encoder.onnx`)로 변환합니다.
 
  - ONNX export 시 `opset_version=14`를 사용하여 지원되지 않는 연산 문제를 해결합니다.

## Notes 
 
- **ONNX Opset Version** :
`extract.py`에서 ONNX 내보내기 시 `opset_version=14`로 설정되어 있어, `aten::scaled_dot_product_attention`와 같은 연산이 지원됩니다.
 
- **동적 입력 크기** :
현재 코드는 480×480 입력 이미지를 기준으로 변환하도록 되어 있습니다. 동적 크기를 지원하려면 관련 전처리 및 Unflatten 처리를 수정해야 합니다.
 
- **모듈 Import** :
langSeg 폴더 내부의 모듈들은 `__init__.py` 파일을 통해 패키지로 인식되며, extract.py에서는 `from langSeg.lseg_module import LSegModule`과 같이 절대 경로로 임포트합니다.

## License 

(프로젝트 라이선스를 여기에 기재하세요. 예: MIT License)


```yaml
---
