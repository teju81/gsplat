import argparse
import torch
import numpy as np
from torch import nn
import os
import json
from plyfile import PlyData, PlyElement
from gsplat.rendering import rasterization
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, assert_never
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from torchvision.transforms import ToTensor
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        #opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = np.asarray(plydata.elements[0]["opacity"])  # shape: [N]
        

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        N = self._xyz.size()[0]

        quats = torch.rand((N, 4), device="cuda")
        self._quats = torch.nn.Parameter(quats)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self._xyz  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self._quats  # [N, 4]
        scales = torch.exp(self._scaling)  # [N, 3]
        opacities = torch.sigmoid(self._opacity)  # [N,]

        image_ids = kwargs.pop("image_ids", None)

        colors = torch.cat((self._features_dc, self._features_rest), 1)  # [N, K, 3]
        rasterize_mode = "classic"
        camera_model = 'pinhole'

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model=camera_model,
            with_ut=False,
            with_eval3d=False,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info


# def normalize_depth(depth, max_depth=5.0):
#     """Normalize depth for visualization"""
#     depth = np.clip(depth, 0, max_depth)
#     depth_vis = (depth / max_depth * 255).astype(np.uint8)
#     return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth_min = depth.min()
    depth_max = depth.max()
    norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    depth_vis = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

def render_rgbd_from_obj(obj_path, cam_intrinsics, cam_pose, width, height):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh from {obj_path}")

    mesh.compute_vertex_normals()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0, 0, 0, 1])

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, material)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, *cam_intrinsics)

    # extrinsic = np.linalg.inv(cam_pose.cpu().numpy()).astype(np.float32)
    # renderer.setup_camera(intrinsic, extrinsic)
    
    extrinsic = np.linalg.inv(cam_pose.squeeze(0).cpu().numpy()).astype(np.float64)
    renderer.setup_camera(intrinsic, extrinsic)

    color = renderer.render_to_image()
    depth = renderer.render_to_depth_image(z_in_view_space=False)

    color_np = np.asarray(color)
    depth_np = np.asarray(depth)

    return color_np, depth_np

def pose_to_camtoworld(tx, ty, tz, qx, qy, qz, qw):
    # Convert quaternion to rotation matrix
    r = R.from_quat([qx, qy, qz, qw])
    R_mat = torch.tensor(r.as_matrix(), dtype=torch.float32)
    t_vec = torch.tensor([tx, ty, tz], dtype=torch.float32)

    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R_mat.T  # inverse rotation
    T[:3, 3] = (-R_mat.T @ t_vec)  # inverse translation
    return T.unsqueeze(0).to("cuda")  # shape: [1, 4, 4]

def main():
    # parser = argparse.ArgumentParser(description="Load trained Gaussian Splat stored as a ply file and render RGBD images.")
    # parser.add_argument("--ply_path", type=str, help="Path to ply file")
    # args = parser.parse_args()

    # # R: 3x3 rotation, t: 3x1 translation
    # R = torch.eye(3).to("cuda")
    # t = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
    # T = torch.eye(4, device="cuda")
    # T[:3, :3] = R.T  # Transpose of R
    # T[:3, 3] = (-R.T @ t).flatten()
    # camtoworlds = T.unsqueeze(0)  # Shape: [1, 4, 4]

    H, W = 1080, 1920
    fx = fy = 1080
    cx, cy = W/2, H/2
    Ks = torch.tensor([[fx, 0.0, cx],
           [0.0, fy, cy],
           [0.0,  0.0,  1.0]], dtype=torch.float32).to("cuda")
    Ks = Ks.unsqueeze(0)  # shape: [1, 3, 3]


    image_ids = torch.tensor([0], dtype=torch.long)  # Shape: [1]
    masks = torch.ones((1, H, W, 4), dtype=torch.bool)  # Shape: [1, 1080, 1920, 4]

    ply_path="/root/code/datasets/ARTGarage/xgrids/4/Gaussian/PLY_Generic_splats_format/point_cloud/iteration_100/point_cloud.ply"

    gaussian_model = GaussianModel(3)

    # Load configuration
    gaussian_model.load_ply(ply_path)

    # Load trajectory
    pose_file =  "panoramicPoses.csv" # "img_traj.csv" "poses.csv"

    img_traj_path = f"/root/code/datasets/ARTGarage/xgrids/1/ResultDataArtGarage_sample_2025-07-17-121502_0/ArtGarage_sample_2025-07-17-121502/{pose_file}"
    df = pd.read_csv(img_traj_path, comment="#", sep='\s+',
                 names=["timestamp", "imgname", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

    # df = pd.read_csv(img_traj_path, comment="#", sep='\s+',
    #              names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

    # Define output path
    video_path = "/root/code/datasets/ARTGarage/xgrids/rendered_comparison.mp4"

    # Define video writer (assumes 1920x1080 images → adjust if needed)
    frame_h, frame_w = H, W
    output_size = (frame_w * 2, frame_h * 2)  # side-by-side: RGB | Depth
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps=0.5
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, output_size)

    for idx, row in df.iterrows():
        if idx > 30:
            break
        # imgname = row["imgname"][:-4]
        # img_path = f"/root/code/datasets/ARTGarage/xgrids/1/ResultDataArtGarage_sample_2025-07-17-121502_0/ArtGarage_sample_2025-07-17-121502/perspective/images/{imgname}_2.jpg"
        # gt_img = cv2.imread(img_path)
        # gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        # gt_tensor = ToTensor()(gt_img_rgb).permute(1, 2, 0).unsqueeze(0).to("cuda")  # [1, H, W, 3]

        # Camera pose
        camtoworlds = pose_to_camtoworld(row.tx, row.ty, row.tz, row.qx, row.qy, row.qz, row.qw)


        # 3DGS Renderings
        renders, alphas, info = gaussian_model.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=3,
            near_plane=0.01,
            far_plane=10000000000.0,
            image_ids=image_ids,
            render_mode="RGB+ED",
            masks=masks,
        )
        colors, depths = renders[..., 0:3], renders[..., 3:4]

        # # Convert to CPU and numpy
        rendered_rgb_3dgs = colors[0].clamp(0, 1).detach().cpu().numpy()  # [H, W, 3]
        rendered_depth_3dgs = depths[0].detach().cpu().numpy()  # [H, W]

        # === Convert to displayable format ===
        rgb_vis_3dgs = (rendered_rgb_3dgs * 255).astype(np.uint8)
        rgb_vis_3dgs = cv2.cvtColor(rgb_vis_3dgs, cv2.COLOR_RGB2BGR)
        depth_vis_3dgs = normalize_depth(rendered_depth_3dgs)
        

        # OBJ File Renderings
        obj_file = "/root/code/datasets/ARTGarage/xgrids/4/Gaussian/Mesh_Files/art_garage_sample.obj"
        #obj_file = "/root/code/datasets/ARTGarage/xgrids/4/Mesh_textured/texture/block0.obj"
        cam_intrinsics = [fx, fy, cx, cy]
        rgb_obj, depth_obj = render_rgbd_from_obj(obj_file, cam_intrinsics, camtoworlds, W, H)
    
        # Ensure OBJ RGB is BGR
        rgb_vis_obj = cv2.cvtColor(rgb_obj, cv2.COLOR_RGB2BGR)
        # Normalize OBJ depth for visualization
        depth_vis_obj = normalize_depth(depth_obj)



        cv2.putText(rgb_vis_3dgs, "GS RGB", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(rgb_vis_obj,  "OBJ RGB", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(depth_vis_3dgs, "GS Depth", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(depth_vis_obj,  "OBJ Depth", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Top row: RGB images
        top_row = np.hstack([rgb_vis_3dgs, rgb_vis_obj])

        # Bottom row: Depth images
        bottom_row = np.hstack([depth_vis_3dgs, depth_vis_obj])

        # Final 2x2 grid
        grid_frame = np.vstack([top_row, bottom_row])  # [2H, 2W, 3]

        video_writer.write(grid_frame)


        # # Show images
        # plt.subplot(2, 2, 1)
        # plt.imshow(rgb_vis_3dgs)
        # plt.title("Rendered RGB 3DGS")


        # plt.subplot(2, 2, 2)
        # plt.imshow(depth_vis_3dgs, cmap='inferno')
        # plt.title("Rendered Depth 3DGS")

        # plt.subplot(2, 2, 3)
        # plt.imshow(rgb_vis_obj)
        # plt.title("GT RGB")

        # plt.subplot(2, 2, 3)
        # plt.imshow(gt_img_rgb)
        # plt.title("OBJ RGB")

        # plt.subplot(2, 2, 4)
        # plt.imshow(depth_vis_obj, cmap='inferno')
        # plt.title("Rendered Depth OBJ")

        # plt.show()


    video_writer.release()
    print(f"✅ Video saved at: {video_path}")

if __name__ == "__main__":
    main()
