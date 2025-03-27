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
from modules.lseg_module import LSegModule
import re
import pandas as pd

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    times = []
    with torch.no_grad():
        for _ in tqdm(range(10), desc="Warm-up"):
            _ = model(input_tensor)
        
        for _ in tqdm(range(iterations), desc="PyTorch Inference"):
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)

    time_avg = np.mean(times)
    time_std = np.std(times)
    print(f"[RESULT] PyTorch Avg Inference Time: {time_avg:.3f} ms ± {time_std:.3f} ms")

    # ✅ PyTorch 모델을 CPU로 이동 (메모리 정리)
    model.to("cpu")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return time_avg, time_std

def measure_onnx_inference_time(onnx_path, input_tensor, iterations=100):
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
    times = []

    # ✅ 입력 텐서 shape 디버깅
    print(f"[DEBUG] 입력 텐서 이름: {input_name}, Shape: {input_array.shape}")

    for _ in tqdm(range(10), desc="Warm-up"):
        try:
            _ = session.run(None, {input_name: input_array})
        except Exception as e:
            print(f"[ERROR] ONNX Warm-up 실행 중 오류 발생: {e}")
            return

    for _ in tqdm(range(iterations), desc="ONNX Inference"):
        start = time.time()
        _ = session.run(None, {input_name: input_array})
        end = time.time()
        times.append((end - start) * 1000)
    
    time_avg = np.mean(times)
    time_std = np.std(times)
    print(f"[RESULT] ONNX Avg Inference Time: {time_avg:.3f} ms ± {time_std:.3f} ms")
    return time_avg, time_std



def measure_tensorrt_inference_time(trt_engine_path, input_tensor, iterations=100):
    print("[INFO] TensorRT (Python) 모델 추론 시작...")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_shape = input_tensor.shape
    output_shape = (1, 512, input_shape[2], input_shape[3])
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

    times = []
    for _ in tqdm(range(iterations), desc="TensorRT Inference"):
        start = time.time()
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    time_avg = np.mean(times)
    time_std = np.std(times)
    print(f"[RESULT] TensorRT (Python) Avg Inference Time: {time_avg:.3f} ms ± {time_std:.3f} ms")
    return time_avg, time_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_sizes", nargs='+', type=int, default=[256, 320, 384, 480], help="List of input image sizes")
    parser.add_argument("--weights", type=str, default="modules/demo_e200.ckpt", help="Path to checkpoint")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    
    # ✅ 체크포인트 경로 (데이터셋 관련 매개변수는 필요 없으므로 제거 가능)
    checkpoint_path = args.weights

    # ✅ 체크포인트 경로에서 태그 추출 (예: demo_e200.ckpt → ade20k, fss_l16.ckpt → fss)
    checkpoint_filename = os.path.basename(checkpoint_path)
    if "ade20k" in checkpoint_filename or "demo" in checkpoint_filename:
        tag = "ade20k"
    elif "fss" in checkpoint_filename:
        tag = "fss"
    else:
        tag = "custom"
    
    for img_size in args.img_sizes:
        print(f"[INFO] Testing image size: {img_size}")
        onnx_path = f"output/models/lseg_img_enc_vit_{img_size}_{tag}.onnx"
        trt_path = f"output/models/lseg_img_enc_vit_{img_size}_{tag}.trt"
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        if not os.path.exists(onnx_path):
            run_subprocess(["python3", "conversion/model_to_onnx.py", "--img_size", str(img_size), "--weights", checkpoint_path])
        if not os.path.exists(trt_path):
            run_subprocess(["python3", "conversion/onnx_to_trt.py", "--img_size", str(img_size), "--weights", checkpoint_path])

        print("[INFO] PyTorch 모델 로드 중...")
        model = LSegModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            backbone="clip_vitl16_384",
            aux=False,
            num_features=256,
            crop_size=img_size,
            readout="project",
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            ignore_index=255,
            dropout=0.0,
            scale_inv=False,
            augment=False,
            no_batchnorm=False,
            widehead=True,
            widehead_hr=False,
            map_location=device,
            arch_option=0,
            block_depth=0,
            activation="lrelu"
        ).net
        
        pt_avg, pt_std = measure_pytorch_inference_time(model, dummy_input, args.iterations)
        #onnx_avg, onnx_std = measure_onnx_inference_time(onnx_path, dummy_input, 1000)
        trt_py_avg, trt_py_std = measure_tensorrt_inference_time(trt_path, dummy_input, args.iterations)
        # Run C++ TensorRT benchmark
        cpp_cmd = [
            "./build/trt_cpp_infer_time_tester",
            trt_path,
            str(args.iterations),
            str(img_size)
        ]
        print(f"[INFO] Running C++ TensorRT benchmark: {' '.join(cpp_cmd)}")
        cpp_proc = subprocess.run(cpp_cmd, capture_output=True, text=True, check=True)

        # Parse C++ stdout
        match = re.search(r"Avg=([\d\.]+) ms ± ([\d\.]+) ms", cpp_proc.stdout)
        cpp_avg, cpp_std = map(float, match.groups())
        print(f"[INFO] TRT C++ Avg: {cpp_avg:.3f} ms ± {cpp_std:.3f} ms")

        results.append({
            "Size": img_size,
            "PyTorch (ms)": pt_avg,
            "PyTorch ±": pt_std,
            "TRT Python (ms)": trt_py_avg,
            "TRT Python ±": trt_py_std,
            "TRT C++ (ms)": cpp_avg,
            "TRT C++ ±": cpp_std,
        })
        
    # Build DataFrame & print
    df = pd.DataFrame(results).set_index("Size")
    print("\n===== Inference Benchmark Summary =====")
    print(df.to_string())
    print("\n===== Inference Benchmark Summary =====")
    print(df.to_markdown())
    
    