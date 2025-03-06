# LSeg Image Encoder ONNX Extraction

이 프로젝트는 LSeg 모델의 이미지 인코딩 경로(백본, 중간 feature 추출, projection layer)를 분리하여 ONNX 모델로 변환하는 코드입니다.
원본 LSeg 구조와 가중치(ckpt)는 그대로 유지하며, 추론용 ONNX 모델을 생성할 수 있도록 구성되었습니다.

---

## **Installation**

### **1. Python 환경**
- Python 3.8 이상 권장

### **2. 필수 라이브러리 설치**
아래 명령어를 사용하여 필요한 라이브러리를 설치하세요:

```bash
pip install -r requirements.txt
```

필요에 따라 가상환경(venv, conda 등)을 사용하세요.

### **3. 모델 다운로드**
대용량 모델 파일은 [Hugging Face](https://huggingface.co/joonyeol99/LSeg_ViT-to-ONNX)에서 다운로드할 수 있습니다.

```bash
wget https://huggingface.co/joonyeol99/LSeg_ViT-to-ONNX/resolve/main/lseg_image_encoder.onnx
wget https://huggingface.co/joonyeol99/LSeg_ViT-to-ONNX/resolve/main/demo_e200.ckpt
```

또는 `download_from_hf.py` 스크립트를 실행하면 자동으로 다운로드됩니다:
```bash
python download_from_hf.py
```

---

## **Project Structure**

```
.
├── demo_e200.ckpt          # 모델 체크포인트 (Hugging Face에서 다운로드 가능)
├── extract.py              # ONNX 변환을 실행하는 스크립트
├── ONNX_to_TRT.py          # ONNX를 TensorRT로 변환하는 코드
├── lseg_image_encoder.onnx # ONNX 변환된 결과물
├── download_from_hf.py     # Hugging Face에서 모델 다운로드하는 스크립트
├── upload_to_hf.py         # Hugging Face에 모델 업로드하는 스크립트
└── langSeg                 # LSeg 관련 모듈 폴더
    ├── __init__.py         # 패키지 인식용 빈 파일
    ├── lseg_blocks.py      # LSeg 블록 및 백본 관련 함수 (경량화된 버전)
    ├── lseg_module.py      # 모델 로드 및 래핑 모듈
    ├── lseg_net.py         # LSegNet 정의 (이미지 인코딩 경로만 포함)
    └── lseg_vit.py         # CLIP 기반 비주얼 백본 및 관련 함수
```

---

## **Usage**

### **1. ONNX 모델 변환 (Export)**
아래 명령어를 실행하면 `demo_e200.ckpt` 체크포인트를 기반으로 ONNX 모델을 생성합니다:

```bash
python3 extract.py
```

이 스크립트는 다음 과정을 수행합니다:
- `demo_e200.ckpt`에서 가중치를 로드하여 모델을 복원합니다.
- 모델의 이미지 인코딩 경로(백본 → 중간 feature 추출 → projection layer)를 분리하여 `LSegImageEncoder` 모듈을 생성합니다.
- 입력 크기 `(1, 3, 480, 480)`의 더미 텐서를 사용하여 `lseg_image_encoder.onnx` 파일로 변환합니다.
- `opset_version=14`로 ONNX export를 수행하여 최신 ONNX 연산을 지원합니다.

**주의:** `lseg_image_encoder.onnx`는 이미 Hugging Face에서 제공되므로, 직접 실행하지 않고 다운로드하여 사용할 수도 있습니다.

### **2. TensorRT 변환**
ONNX 모델을 TensorRT 엔진으로 변환하려면:
```bash
python3 ONNX_to_TRT.py
```
**주의:** TensorRT 변환은 GPU 및 환경에 따라 다르므로, 실행할 기기에서 직접 변환해야 합니다.

---

## **Additional Notes**

### **1. ONNX Opset Version**
- `extract.py`는 `opset_version=14`을 사용하여 `aten::scaled_dot_product_attention` 등의 최신 연산을 지원하도록 설정되어 있습니다.

### **2. 입력 크기**
- 현재 ONNX 모델은 `480×480` 입력 크기를 기준으로 변환됩니다.
- 동적 입력 크기를 지원하려면 `extract.py`의 `torch.onnx.export()` 설정을 수정해야 합니다.

### **3. 모듈 Import**
- `langSeg` 폴더 내부의 모듈들은 `__init__.py`를 통해 패키지로 인식됩니다.
- `extract.py`에서 `from langSeg.lseg_module import LSegModule`과 같은 절대 경로로 import 가능합니다.

---

## **License**
(사용하고 있는 라이선스를 여기에 기재하세요. 예: MIT License)

---
