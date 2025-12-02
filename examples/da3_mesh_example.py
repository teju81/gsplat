import os
import json
import glob
import numpy as np
import open3d as o3d
import imageio
from PIL import Image
from tqdm import tqdm
import torch

from depth_anything_3.api import DepthAnything3

#########################################
# CONFIG
#########################################

base_folder = "/root/code/ubuntu_data/code_outputs/warehouse_drone_trajectory"

color_dir  = os.path.join(base_folder, "color")
poses_json = os.path.join(base_folder, "poses.json")

output_depth_dir = os.path.join(base_folder, "depth_da3")
mesh_output_path = os.path.join(base_folder, "mesh.ply")

os.makedirs(output_depth_dir, exist_ok=True)

#########################################
# LOAD DepthAnything3
#########################################

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

#########################################
# LOAD pose json
#########################################

with open(poses_json, "r") as f:
    pose_dict = json.load(f)["poses"]

pose_keys = sorted(pose_dict.keys(), key=lambda x: int(x))
pose_list = [np.array(pose_dict[k], dtype=np.float32) for k in pose_keys]

print(f"Loaded {len(pose_list)} poses")

#########################################
# TSDF FUSION SETUP
#########################################

voxel_length = 0.01
sdf_trunc    = 0.04
max_depth    = 8.0

tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_length,
    sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

#########################################
# LOAD RGB
#########################################

rgb_paths = sorted(glob.glob(os.path.join(color_dir, "*.png")))
print(rgb_paths)
N = len(rgb_paths)
#assert N == len(pose_list)

#########################################
# PROCESS IMAGES
#########################################

for i, rgb_path in enumerate(tqdm(rgb_paths)):
    if i > 100:
        break

    rgb_np = imageio.imread(rgb_path)
    pil_img = Image.fromarray(rgb_np)

    pred = model.inference([pil_img])
    depth = pred.depth[0].astype(np.float32)

    # --- FIX 3: Handle invalid depth values ---
    depth[depth <= 0] = np.nan

    # Save depth
    exr_path = os.path.join(output_depth_dir, f"depth_{i:04d}.exr")
    imageio.imwrite(exr_path, depth)


    # Save PNG (optional)
    png_path = os.path.join(output_depth_dir, f"depth_{i:04d}.png")
    depth_mm = (depth * 1000).astype(np.uint16)
    imageio.imwrite(png_path, depth_mm)

    # Load pose
    c2w = pose_list[i]
    w2c = np.linalg.inv(c2w)

    # --- FIX 1: Resize RGB to match DA3 depth resolution ---
    H, W = depth.shape
    rgb_np_resized = np.array(Image.fromarray(rgb_np).resize((W, H)))

    # --- FIX 2: Intrinsics must match DA3 output ---
    K = pred.intrinsics[0]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    color_o3d = o3d.geometry.Image(rgb_np_resized.astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False
    )

    tsdf_volume.integrate(rgbd, intrinsics_o3d, w2c)

#########################################
# EXTRACT MESH
#########################################

mesh = tsdf_volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(mesh_output_path, mesh)

print(f"\nMesh saved → {mesh_output_path}")
