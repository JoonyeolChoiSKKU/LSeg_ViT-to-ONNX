import argparse
import torch
import torch.nn as nn
import torch.onnx
from lseg.lseg_module import LSegModule
from lseg.image_encoder import LSegImageEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=480, help="Input image size")
    args = parser.parse_args()
    
    checkpoint_path = "models/demo_e200.ckpt"
    lseg_module = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
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
    image_encoder.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    onnx_filename = f"models/lseg_image_encoder_{args.img_size}.onnx"
    
    torch.onnx.export(
        image_encoder,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes=None
    )
    
    print(f"ONNX export complete: {onnx_filename}")
