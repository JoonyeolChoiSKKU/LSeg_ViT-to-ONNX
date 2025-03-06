# LSeg Image Encoder ONNX & TensorRT

이 프로젝트는 **LSeg 모델의 이미지 인코딩 경로(백본, 중간 feature 추출, projection layer)**를 분리하여 ONNX 및 TensorRT 모델로 변환하는 코드입니다.  
**PyTorch → ONNX → TensorRT 변환 및 Inference 성능 비교**를 지원합니다.

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
**3. APT 패키지 설치 (필요한 경우)** 
일부 패키지는 시스템 라이브러리가 필요하므로 아래 명령어를 실행하세요:


```bash
sudo apt update && sudo apt install -y \
    python3-pip python3-dev \
    libopencv-dev \
    libprotobuf-dev protobuf-compiler \
    libtinfo5 \
    libopenmpi-dev
```
**4. 모델 다운로드** 대용량 모델 파일은 [Hugging Face](https://huggingface.co/joonyeol99/LSeg_ViT-to-ONNX) 에서 다운로드할 수 있습니다.

```bash
wget https://huggingface.co/joonyeol99/LSeg_ViT-to-ONNX/resolve/main/lseg_image_encoder.onnx
wget https://huggingface.co/joonyeol99/LSeg_ViT-to-ONNX/resolve/main/demo_e200.ckpt
```
또는 `download_from_hf.py` 스크립트를 실행하면 자동으로 다운로드됩니다:

```bash
python3 models/srcipts/download_from_hf.py
```


---

**Project Structure** 

```bash
.
├── models/                   # 변환된 모델 및 체크포인트 저장 폴더
│   ├── demo_e200.ckpt        # LSeg 원본 체크포인트
│   ├── lseg_image_encoder.onnx # 변환된 ONNX 모델
│   ├── lseg_image_encoder.trt  # 변환된 TensorRT 모델
│
├── conversion/               # 모델 변환 관련 코드
│   ├── model_to_onnx.py      # PyTorch → ONNX 변환 스크립트
│   ├── onnx_to_trt.py        # ONNX → TensorRT 변환 스크립트
│
├── inference/                # 추론 및 성능 비교 코드
│   ├── inferenceTimeTester.py # PyTorch, ONNX, TensorRT Inference 성능 비교
│
├── lseg/                     # LSeg 관련 모듈
│   ├── __init__.py
│   ├── image_encoder.py      # LSeg 이미지 인코더 (PyTorch 모델 래핑)
│   ├── lseg_blocks.py
│   ├── lseg_module.py
│   ├── lseg_net.py
│   ├── lseg_vit.py
│
├── scripts/                  # 데이터 다운로드 및 업로드 스크립트
│   ├── download_from_hf.py
│   ├── upload_to_hf.py
│
├── requirements.txt          # 필요한 Python 패키지 목록
└── README.md                 # 프로젝트 설명 파일
```


---

**Usage** **1. ONNX 모델 변환 (Export)** 아래 명령어를 실행하면 `demo_e200.ckpt` 체크포인트를 기반으로 ONNX 모델을 생성합니다:

```bash
python3 conversion/model_to_onnx.py
```

이 스크립트는 다음 과정을 수행합니다:
 
- `demo_e200.ckpt`에서 가중치를 로드하여 모델을 복원합니다.
 
- 모델의 이미지 인코딩 경로(백본 → 중간 feature 추출 → projection layer)를 분리하여 `LSegImageEncoder` 모듈을 생성합니다.
 
- 입력 크기 `(1, 3, 480, 480)`의 더미 텐서를 사용하여 `lseg_image_encoder.onnx` 파일로 변환합니다.
**주의:**  `lseg_image_encoder.onnx`는 이미 Hugging Face에서 제공되므로, 직접 실행하지 않고 다운로드하여 사용할 수도 있습니다.**2. TensorRT 변환** 
ONNX 모델을 TensorRT 엔진으로 변환하려면:


```bash
python3 conversion/onnx_to_trt.py
```
**주의:**  TensorRT 변환은 GPU 및 환경에 따라 다르므로, 실행할 기기에서 직접 변환해야 합니다.

---

**3. 추론 및 성능 비교** **자동 실행: PyTorch vs ONNX vs TensorRT Inference** 아래 명령어를 실행하면 **PyTorch, ONNX, TensorRT Inference 속도를 비교** 할 수 있습니다:

```bash
python3 inferenceTimeTester.py
```

이 스크립트는 다음 과정을 자동으로 수행합니다:
 
1. `demo_e200.ckpt` 체크포인트가 없으면 **자동 다운로드**
 
2. `lseg_image_encoder.onnx`가 없으면 **자동 변환**
 
3. `lseg_image_encoder.trt`가 없으면 **자동 변환**
 
4. **추론 속도 비교**  (PyTorch → ONNX → TensorRT)
**추론 결과 예시** 

```less
[INFO] PyTorch 모델 추론 (GPU) 시작...
[RESULT] PyTorch Model Inference Time: 0.143024 sec (GPU)

[INFO] ONNX 모델 추론 (GPU) 시작...
[RESULT] ONNX Model Inference Time: 1.649300 sec (CPU)  # GPU 비활성화 시 느려짐

[INFO] TensorRT 모델 추론 시작...
[RESULT] TensorRT Model Inference Time: 0.123930 sec
```
🔥 **TensorRT가 가장 빠른 속도를 보임!** 
→ **ONNX Runtime은 CUDA 설정이 필요하며, GPU에서 실행해야 빠름** 

---

**Additional Notes** **1. ONNX Opset Version**  
- `model_to_onnx.py`는 `opset_version=14`을 사용하여 최신 연산을 지원합니다.
**2. 입력 크기**  
- 현재 ONNX 모델은 `480×480` 입력 크기를 기준으로 변환됩니다.
 
- 동적 입력 크기를 지원하려면 `torch.onnx.export()` 설정을 수정해야 합니다.
**3. GPU 설정 확인**  
- PyTorch, ONNX Runtime, TensorRT가 **모두 GPU에서 실행되는지 확인하는 것이 중요** 합니다.
 
- `onnxruntime-gpu`가 필요할 경우 아래 명령어로 설치하세요:


```bash
pip3 install onnxruntime-gpu
```
 
- `CUDAExecutionProvider`가 활성화되었는지 확인하려면:


```python
import onnxruntime as ort
print(ort.get_available_providers())
```


---

**License** 
(사용하고 있는 라이선스를 여기에 기재하세요. 예: MIT License)


