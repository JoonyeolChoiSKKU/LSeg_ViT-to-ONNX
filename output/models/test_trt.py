import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 자동으로 CUDA 초기화
import numpy as np

# TensorRT 로거 생성
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

# 엔진 로드 (엔진 파일 경로 수정)
engine_path = "lseg_image_encoder_384.trt"
engine = load_engine(engine_path)
context = engine.create_execution_context()

def get_binding_index_by_name(engine, binding_name):
    n = engine.num_bindings()  # 함수 호출 방식으로 변경
    for i in range(n):
        if engine.get_binding_name(i) == binding_name:
            return i
    raise ValueError(f"Binding name {binding_name} not found in engine.")

input_binding_idx = get_binding_index_by_name(engine, "input")
output_binding_idx = get_binding_index_by_name(engine, "output")

print("Input binding shape:", engine.get_binding_shape(input_binding_idx))
print("Output binding shape:", engine.get_binding_shape(output_binding_idx))

# 입력 및 출력 메모리 할당
input_shape = engine.get_binding_shape(input_binding_idx)
output_shape = engine.get_binding_shape(output_binding_idx)
input_size = trt.volume(input_shape) * np.float32().itemsize
output_size = trt.volume(output_shape) * np.float32().itemsize

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# 더미 입력 생성 (예: 1x3x384x384)
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# 입력 데이터를 GPU 메모리로 복사 후 추론 실행
cuda.memcpy_htod_async(d_input, dummy_input, stream)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

# 출력 데이터를 CPU 메모리로 복사
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output_data, d_output, stream)
stream.synchronize()

print("TensorRT 엔진 추론 출력 shape:", output_data.shape)
print("TensorRT 엔진 추론 출력 (일부):", output_data.flatten()[:10])
