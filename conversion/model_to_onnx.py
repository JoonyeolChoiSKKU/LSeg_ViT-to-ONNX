import argparse
import torch
import torch.nn as nn
import torch.onnx
import os, sys
# ── 스크립트 위치의 상위 폴더(=프로젝트 루트)를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.lseg_module import LSegModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=480, help="Input image size")
    parser.add_argument("--weights", type=str, default="modules/demo_e200.ckpt", help="Path to checkpoint")
    args = parser.parse_args()
    # dynamic H×W 지원하더라도 내부 crop_size 용으로 한 변 크기를 만들어 둡니다.
    args.crop_size = args.img_size
    # ✅ 디바이스 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

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

    model = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone="clip_vitl16_384",
        aux=False,
        num_features=256,
        crop_size=args.crop_size,
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
        map_location="cpu",
        #map_location=device,
        arch_option=0,
        block_depth=0,
        activation="lrelu"
    ).net
    # ).net.to(device)

    model.eval()

    #dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    onnx_filename = f"output/models/lseg_img_enc_vit_{tag}.onnx"
    
    torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    input_names=["input"],
    output_names=["output"],
    opset_version=14,
    # ← 여기서 dynamic_axes 지정
    dynamic_axes={
        "input":  {2: "height", 3: "width"},
        "output": {2: "height", 3: "width"},
    }
)
print(f"✅ Dynamic ONNX 저장: {onnx_filename}")
