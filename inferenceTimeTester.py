import os
import torch
import onnxruntime as ort
import tensorrt as trt
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
import subprocess
import argparse
from tqdm import tqdm
from lseg.lseg_module import LSegModule
from lseg.image_encoder import LSegImageEncoder

def run_subprocess(command):
    try:
        print(f"[INFO] 실행 중: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"[INFO] 실행 완료: {' '.join(command)}")
    except subprocess.CalledProcessError:
        print(f"[ERROR] 실행 중 오류 발생: {' '.join(command)}")
        exit(1)

def measure_pytorch_inference_time(model, input_tensor, iterations=100):
    print("[INFO] PyTorch 모델 추론 (GPU) 시작...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        for _ in tqdm(range(10), desc="Warm-up"):
            _ = model(input_tensor)
        
        start_time = time.time()
        for _ in tqdm(range(iterations), desc="PyTorch Inference"):
            _ = model(input_tensor)
        end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"[RESULT] PyTorch Model Inference Time: {(avg_time * 1000):.6f} ms (GPU)")

    # ✅ PyTorch 모델을 CPU로 이동 (메모리 정리)
    model.to("cpu")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def measure_onnx_inference_time(onnx_path, input_tensor, iterations=100):
    print("[INFO] ONNX 모델 추론 (CPU) 테스트 중...")

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_array = input_tensor.cpu().numpy()

    for _ in range(10):
        _ = session.run(None, {input_name: input_array})

    print("[INFO] ONNX Inference CPU 실행 성공!")

def measure_onnx_inference_time_(onnx_path, input_tensor, iterations=100):
    print("[INFO] ONNX 모델 추론 (GPU) 시작...")

    # ✅ ONNX 모델 파일이 올바르게 존재하는지 확인
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX 파일이 존재하지 않습니다: {onnx_path}")
        return
    
    try:
        session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        print("[INFO] ONNX 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"[ERROR] ONNX 모델 로드 중 오류 발생: {e}")
        return

    input_name = session.get_inputs()[0].name
    input_array = input_tensor.cpu().numpy()  # ✅ GPU에서 CPU로 이동

    # ✅ 입력 텐서 shape 디버깅
    print(f"[DEBUG] 입력 텐서 이름: {input_name}, Shape: {input_array.shape}")

    for _ in tqdm(range(10), desc="Warm-up"):
        try:
            _ = session.run(None, {input_name: input_array})
        except Exception as e:
            print(f"[ERROR] ONNX Warm-up 실행 중 오류 발생: {e}")
            return

    start_time = time.time()
    for _ in tqdm(range(iterations), desc="ONNX Inference"):
        try:
            _ = session.run(None, {input_name: input_array})
        except Exception as e:
            print(f"[ERROR] ONNX Inference 실행 중 오류 발생: {e}")
            return
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"[RESULT] ONNX Model Inference Time: {(avg_time * 1000):.6f} ms (GPU)")



def measure_tensorrt_inference_time(trt_engine_path, input_tensor, iterations=100):
    print("[INFO] TensorRT 모델 추론 시작...")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_shape = input_tensor.shape
    output_shape = (1, 512, input_shape[2] // 2, input_shape[3] // 2)
    d_input = cuda.mem_alloc(input_tensor.nbytes)
    d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(np.float32).itemsize))
    stream = cuda.Stream()
    host_input = np.array(input_tensor.numpy(), dtype=np.float32, order='C')
    host_output = np.empty(output_shape, dtype=np.float32, order='C')
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, input_shape)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    for _ in tqdm(range(10), desc="Warm-up"):
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()
    start_time = time.time()
    for _ in tqdm(range(iterations), desc="TensorRT Inference"):
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"[RESULT] TensorRT Model Inference Time: {(avg_time * 1000):.6f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_sizes", nargs='+', type=int, default=[128, 320, 384, 480], help="List of input image sizes")
    args = parser.parse_args()
    
    checkpoint_path = "models/demo_e200.ckpt"
    print("[INFO] PyTorch 모델 로드 중...")
    lseg_module = LSegModule.load_from_checkpoint(checkpoint_path=checkpoint_path)
    lseg_net = lseg_module.net if hasattr(lseg_module, 'net') else lseg_module
    image_encoder = LSegImageEncoder(lseg_net)
    
    for img_size in args.img_sizes:
        print(f"[INFO] Testing image size: {img_size}")
        onnx_path = f"models/lseg_image_encoder_{img_size}.onnx"
        trt_path = f"models/lseg_image_encoder_{img_size}.trt"
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        if not os.path.exists(onnx_path):
            run_subprocess(["python3", "conversion/model_to_onnx.py", "--img_size", str(img_size)])
        if not os.path.exists(trt_path):
            run_subprocess(["python3", "conversion/onnx_to_trt_optimized.py", "--img_size", str(img_size)])
        
        measure_pytorch_inference_time(image_encoder, dummy_input)
        measure_onnx_inference_time(onnx_path, dummy_input)
        measure_tensorrt_inference_time(trt_path, dummy_input)