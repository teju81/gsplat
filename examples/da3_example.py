import glob, os, torch
import numpy as np
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

input_dir = "/root/code/ubuntu_data/code_outputs/second_floor_trajectory/color"
output_dir = "/root/code/ubuntu_data/code_outputs/second_floor_trajectory/depth_da3"
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

for i, img_path in enumerate(image_paths):
    print(f"Processing {i+1}/{len(image_paths)}: {img_path}")

    # Process ONE image at a time
    prediction = model.inference([img_path])

    depth = prediction.depth[0]        # [H, W]
    conf  = prediction.conf[0]         # [H, W]
    K     = prediction.intrinsics[0]   # [3, 3]
    RT    = prediction.extrinsics[0]   # [3, 4]

    # Save depth as EXR
    depth_path = os.path.join(output_dir, f"depth_{i:04d}.exr")
    import imageio
    imageio.imwrite(depth_path, depth.astype(np.float32))

    # Save PNG (optional)
    png_path = os.path.join(output_dir, f"depth_{i:04d}.png")
    depth_mm = (depth * 1000).astype(np.uint16)
    imageio.imwrite(png_path, depth_mm)
