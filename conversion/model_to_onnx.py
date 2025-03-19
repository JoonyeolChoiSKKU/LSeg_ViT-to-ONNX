import argparse
import torch
import torch.nn as nn
import torch.onnx
from modules.lseg_module import LSegModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=384, help="Input image size")
    args = parser.parse_args()
    # ✅ 디바이스 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    # ✅ 체크포인트 경로 (데이터셋 관련 매개변수는 필요 없으므로 제거 가능)
    checkpoint_path = "modules/demo_e200.ckpt"
    model = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone="clip_vitl16_384",
        aux=False,
        num_features=256,
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
    onnx_filename = f"output/models/lseg_image_encoder_{args.img_size}.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes=None
    )
    
    print(f"✅ Image Encoder가 ONNX 파일로 저장되었습니다: {onnx_filename}")
