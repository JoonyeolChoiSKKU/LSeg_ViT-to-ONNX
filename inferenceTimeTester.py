import os, re
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
import pandas as pd
import torch.nn.functional as F

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
    model.to(device)
    input_tensor = input_tensor.to(device)

    # ▶▶ single-run debug: sum/mean 찍기
    with torch.no_grad():
        out = model(input_tensor)
    out_np = out.cpu().numpy()
    print(f"[DEBUG] PyTorch single-run output sum={out_np.sum():.4f}, mean={out_np.mean():.4f}")

    times = []
    with torch.no_grad():
        for _ in tqdm(range(10), desc="Warm-up PyTorch"):
            _ = model(input_tensor)
        for _ in tqdm(range(iterations), desc="PyTorch Inference"):
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)
    avg, std = np.mean(times), np.std(times)
    print(f"[RESULT] PyTorch Avg Inference Time: {avg:.3f} ms ± {std:.3f} ms")
    model.to("cpu")
    torch.cuda.empty_cache()
    return avg, std


def measure_onnx_inference_time(onnx_path, input_tensor, iterations=100):
    print("[INFO] ONNX 모델 추론 (GPU) 시작...")
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX 파일이 존재하지 않습니다: {onnx_path}")
        return None, None
    try:
        session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        print("[INFO] ONNX 모델 로드 완료")
    except Exception as e:
        print(f"[ERROR] ONNX 모델 로드 중 오류: {e}")
        return None, None
    input_name = session.get_inputs()[0].name
    input_array = input_tensor.cpu().numpy().astype(np.float32)
    for _ in range(10):
        session.run(None, {input_name: input_array})
    times = []
    for _ in tqdm(range(iterations), desc="ONNX Inference"):
        start = time.time()
        session.run(None, {input_name: input_array})
        end = time.time()
        times.append((end - start) * 1000)
    avg, std = np.mean(times), np.std(times)
    print(f"[RESULT] ONNX Avg Inference Time: {avg:.3f} ms ± {std:.3f} ms")
    return avg, std


def measure_tensorrt_inference_time(trt_engine_path, input_tensor, iterations=100, dynamic=False):
    print("[INFO] TensorRT (Python) 모델 추론 시작...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 바인딩 이름 조회
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # dynamic 입력 크기 설정
    if dynamic:
        context.set_input_shape(input_name, tuple(input_tensor.shape))

    # 실제 출력 텐서 shape 조회
    out_dims = context.get_tensor_shape(output_name)
    output_shape = tuple(out_dims)

    # I/O 버퍼 할당 (실제 크기 기반)
    host_input = input_tensor.cpu().numpy().astype(np.float32)
    host_output = np.empty(output_shape, dtype=np.float32, order='C')
    d_input = cuda.mem_alloc(host_input.nbytes)
    d_output = cuda.mem_alloc(host_output.nbytes)

    # 메모리 주소 등록
    context.set_tensor_address(input_name,  int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    stream = cuda.Stream()

    def do_infer():
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()
        return host_output

    # Warm-up
    for _ in tqdm(range(10), desc="Warm-up TRT"):
        do_infer()
    
    # ▶▶▶ 여기서 디버그 한 번 ▶▶▶
    debug_out = do_infer()
    print(f"[DEBUG] TRT single-run output sum={debug_out.sum():.4f}, mean={debug_out.mean():.4f}")


    # Timing
    times = []
    for _ in tqdm(range(iterations), desc="TensorRT Inference"):
        start = time.time()
        _ = do_infer()
        end = time.time()
        times.append((end - start) * 1000)
    avg, std = np.mean(times), np.std(times)
    print(f"[RESULT] TensorRT Avg Inference Time: {avg:.3f} ms ± {std:.3f} ms")
    return avg, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_sizes", nargs='+', type=int,
                        default=[260,390,520,650,780,910],
                        help="List of raw input image sizes (H=W) to test")
    parser.add_argument("--weights", type=str,
                        default="modules/demo_e200.ckpt")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dynamic_trt", type=str,
                        default="output/models/lseg_img_enc_vit_ade20k.trt",
                        help="Path to the single dynamic TRT engine")
    parser.add_argument("--resize", type=int, default=None,
                        help="If set, resize every image to this square size before inference")
    args = parser.parse_args()

    checkpoint_path = args.weights
    ck = os.path.basename(checkpoint_path)
    tag = "ade20k" if "ade20k" in ck or "demo" in ck else "fss" if "fss" in ck else "custom"

    # ONNX / TRT 파일 생성 (없으면 자동)
    onnx_path = f"output/models/lseg_img_enc_vit_{tag}.onnx"
    trt_path  = args.dynamic_trt
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    if not os.path.exists(onnx_path):
        run_subprocess(["python3", "conversion/model_to_onnx.py", "--img_size", str(args.img_sizes[0]), "--weights", checkpoint_path])
    if not os.path.exists(trt_path):
        run_subprocess(["python3", "conversion/onnx_to_trt_dynamic.py", "--onnx", onnx_path, "--engine", trt_path])

    for img_size in args.img_sizes:
        print(f"\n\n[INFO] Testing image size: {img_size}\n\n")
        # ▶ 1.0 으로 채운 입력 (모든 채널・픽셀이 1.0)
        dummy_input = torch.ones(1,3,img_size,img_size, dtype=torch.float32)
        # inp = F.interpolate(dummy_input, size=(args.resize,args.resize), mode='bilinear', align_corners=False) if args.resize else dummy_input
        if args.resize:
            # 정사각형 리사이즈 후에도 모두 1.0
            inp = torch.ones(1,3,args.resize,args.resize, dtype=torch.float32)
        else:
            inp = dummy_input
        trt_py_avg, trt_py_std = measure_tensorrt_inference_time(trt_path, inp, args.iterations, dynamic=True)
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


        ################################################################################
        ###################################    Inference   #############################
        ################################################################################
        pt_avg, pt_std = measure_pytorch_inference_time(model, dummy_input, args.iterations)
        # onnx_avg, onnx_std = measure_onnx_inference_time(onnx_path, inp, args.iterations)
        # ▶ 이제 img_h, img_w 두 인자를 넘겨줍니다
        cpp_cmd = [
            "./build/trt_cpp_infer_time_tester",
            trt_path,
            str(args.iterations),
            # resize 옵션이 있으면 그 값을, 없으면 원본 img_size 사용
            str(args.resize or img_size),
            str(args.resize or img_size)
        ]
        print(f"[INFO] Running C++ TensorRT benchmark: {' '.join(cpp_cmd)}")
        # → stdout을 캡처해 변수에 담으면서도 터미널에 그대로 흘려보내기
        cpp_proc = subprocess.run(
            cpp_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        # 1) 터미널에 찍기 (TensorRT 로그 줄은 건너뛰기)
        for line in cpp_proc.stdout.splitlines():
            if not line.startswith("[TensorRT]"):
                print(line)

        # 2) 파싱해서 table에 넣기
        cpp_avg = cpp_std = None
        for line in cpp_proc.stdout.splitlines():
            m = re.search(r"Avg=([\d\.]+) ms ± ([\d\.]+) ms", line)
            if m:
                cpp_avg, cpp_std = map(float, m.groups())
                break
        if cpp_avg is None:
            print("[WARN] C++ 결과를 파싱하지 못했습니다.")
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
    df = pd.DataFrame(results).set_index("Size")
    print("\n===== Inference Benchmark Summary =====")
    print(df.to_string())
    print("\n===== Inference Benchmark Summary =====")
    print(df.to_markdown())
