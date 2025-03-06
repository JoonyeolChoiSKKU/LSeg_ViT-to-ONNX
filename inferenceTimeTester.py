import os
import torch
import onnxruntime as ort
import tensorrt as trt
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
import subprocess
from tqdm import tqdm  # 진행 상태 가시화

# 모델 파일 경로 설정
CKPT_PATH = "models/demo_e200.ckpt"
ONNX_PATH = "models/lseg_image_encoder.onnx"
TRT_PATH = "models/lseg_image_encoder.trt"

# Google Drive 파일 ID
GDRIVE_FILE_ID = "1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7"

# 더미 입력 데이터
dummy_input = torch.randn(1, 3, 480, 480)

### 1. 체크포인트 확인 및 다운로드 ###
def download_ckpt():
    print("[INFO] demo_e200.ckpt 파일이 없습니다. 다운로드를 시작합니다...")
    try:
        subprocess.run(["gdown", f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", "-O", CKPT_PATH], check=True)
        print("[INFO] 다운로드 완료: demo_e200.ckpt")
    except subprocess.CalledProcessError:
        print("[ERROR] gdown을 통해 다운로드할 수 없습니다. 직접 다운로드 후 models 폴더에 넣어주세요.")
        exit(1)

### 2. ONNX 변환 ###
def convert_to_onnx():
    print("[INFO] ONNX 모델이 없습니다. model_to_onnx.py 실행하여 변환합니다...")
    try:
        subprocess.run(["python3", "conversion/model_to_onnx.py"], check=True)
        print("[INFO] ONNX 변환 완료")
    except subprocess.CalledProcessError:
        print("[ERROR] ONNX 변환 중 오류 발생")
        exit(1)

### 3. TensorRT 변환 ###
def convert_to_trt():
    print("[INFO] TensorRT 모델이 없습니다. onnx_to_trt.py 실행하여 변환합니다...")
    try:
        subprocess.run(["python3", "conversion/onnx_to_trt.py"], check=True)
        print("[INFO] TensorRT 변환 완료")
    except subprocess.CalledProcessError:
        print("[ERROR] TensorRT 변환 중 오류 발생")
        exit(1)

### 4. PyTorch Inference ###
from lseg.lseg_module import LSegModule
from lseg.image_encoder import LSegImageEncoder

def measure_pytorch_inference_time(model, input_tensor, iterations=100):
    print("[INFO] PyTorch 모델 추론 (GPU) 시작...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for _ in tqdm(range(10), desc="Warm-up"):  # Warm-up
            _ = model(input_tensor)

        start_time = time.time()
        for _ in tqdm(range(iterations), desc="PyTorch Inference"):
            _ = model(input_tensor)
        end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"[RESULT] PyTorch Model Inference Time: {(avg_time*1000):.6f} ms (GPU)")

### 5. ONNX Inference ###
def measure_onnx_inference_time(onnx_path, input_tensor, iterations=100):
    print("[INFO] ONNX 모델 추론 (GPU) 시작...")
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_array = input_tensor.numpy()

    for _ in tqdm(range(10), desc="Warm-up"):  # Warm-up
        _ = session.run(None, {input_name: input_array})

    start_time = time.time()
    for _ in tqdm(range(iterations), desc="ONNX Inference"):
        _ = session.run(None, {input_name: input_array})
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"[RESULT] ONNX Model Inference Time: {(avg_time*1000):.6f} ms (GPU)")


### 6. TensorRT Inference ###
def load_tensorrt_engine(trt_engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

#output_shape = (1, 512, 240, 240)  # 모델의 출력 크기 확인 필요

def measure_tensorrt_inference_time(trt_engine_path, input_tensor, iterations=100):
    print("[INFO] TensorRT 모델 추론 시작...")
    engine = load_tensorrt_engine(trt_engine_path)
    context = engine.create_execution_context()

    input_shape = input_tensor.shape
    output_shape = (1, 512, 240, 240)  # 모델의 출력 크기 확인 필요
    input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)  # int 변환 추가
    output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)  # int 변환 추가

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    stream = cuda.Stream()
    host_input = np.array(input_tensor.numpy(), dtype=np.float32, order='C')
    host_output = np.empty(output_shape, dtype=np.float32, order='C')

    # 입력 및 출력 텐서 이름 가져오기
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    print(f"[INFO] TensorRT 입력 텐서 이름: {input_name}")
    print(f"[INFO] TensorRT 출력 텐서 이름: {output_name}")

    # 입력 및 출력 shape 설정
    context.set_input_shape(input_name, input_shape)
    print(f"[INFO] 입력 shape 설정 완료: {context.get_tensor_shape(input_name)}")

    # 입력과 출력의 메모리 주소 설정 (필수!)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Warm-up 단계
    for _ in tqdm(range(10), desc="Warm-up"):
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()

    # 성능 측정
    start_time = time.time()
    for _ in tqdm(range(iterations), desc="TensorRT Inference"):
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"[RESULT] TensorRT Model Inference Time: {(avg_time*1000):.6f} ms")

### 🔥 실행 부분 ###
if __name__ == "__main__":
    print("🚀 [INFO] LSeg Image Encoder Inference Pipeline 시작 🚀")

    # 1️⃣ 체크포인트 확인 및 다운로드
    if not os.path.exists(CKPT_PATH):
        download_ckpt()

    # 2️⃣ ONNX 모델 변환
    if not os.path.exists(ONNX_PATH):
        convert_to_onnx()

    # 3️⃣ TensorRT 모델 변환
    if not os.path.exists(TRT_PATH):
        convert_to_trt()

    # 4️⃣ PyTorch 모델 로드
    print("[INFO] PyTorch 모델 로드 중...")
    lseg_module = LSegModule.load_from_checkpoint(
        checkpoint_path=CKPT_PATH,
        data_path='../datasets/',
        dataset='ade20k',
        backbone='clip_vitl16_384',
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu'
    )
    lseg_net = lseg_module.net if hasattr(lseg_module, 'net') else lseg_module
    image_encoder = LSegImageEncoder(lseg_net)

    print("📌 [INFO] 추론 시간 측정을 시작합니다.")
    measure_pytorch_inference_time(image_encoder, dummy_input)
    measure_onnx_inference_time(ONNX_PATH, dummy_input)
    measure_tensorrt_inference_time(TRT_PATH, dummy_input)
