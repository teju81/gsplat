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

import pygame
from pygame.locals import *
import math
import glm # We'll use PyGLM for easier matrix math
import sys
import os

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


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        window, img, depth = param  # Unpack color/depth flag
        val = depth[y, x]
        text = f"({x}, {y}): {val:.3f}"
        cv2.putText(img, text, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imshow(window, img)

def quadrant_callback(event, x, y, flags, param):
    rgb1, rgb2, depth1, depth2, depth_3dgs_rescale, depth_obj_rescale = param
    h, w, _ = rgb1.shape
    if y < h and x < w:
        on_mouse_click(event, x, y, flags, ("Renderings", rgb1, depth_3dgs_rescale))
    elif y < h and x >= w:
        on_mouse_click(event, x - w, y, flags, ("Renderings", rgb2, depth_obj_rescale))
    elif y >= h and x < w:
        on_mouse_click(event, x, y - h, flags, ("Renderings", depth1, depth_3dgs_rescale))
    elif y >= h and x >= w:
        on_mouse_click(event, x - w, y - h, flags, ("Renderings", depth2, depth_obj_rescale))
    # re-render grid with updated images
    grid = np.vstack([np.hstack([rgb1, rgb2]), np.hstack([depth1, depth2])])
    cv2.imshow("Renderings", grid)

def normalize_depth(depth: np.ndarray, depth_min=0.0, depth_max=30.0) -> np.ndarray:
    norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    depth_vis = (norm * 255).astype(np.uint8)

    #return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)  # convert to 3-channel BGR for OpenCV annotation

class Recorder:
    def __init__(self, color_dir="color", depth_dir="depth", json_path="/root/code/output/gaussian_splatting/xgrids_vr/poses.json"):
        self.color_dir = color_dir
        self.depth_dir = depth_dir
        self.json_path = json_path
        self.data = []
        self.frame_id = 0

        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

    def record(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray):
        """
        Args:
            rgb (H x W x 3): np.uint8
            depth (H x W): np.float32 or np.uint16
            pose (4 x 4): np.ndarray, camera-to-world matrix
        """
        # File names
        name = f"frame_{self.frame_id:04d}.png"
        color_path = os.path.join(self.color_dir, name)
        depth_path = os.path.join(self.depth_dir, name)

        # Save images
        cv2.imwrite(color_path, rgb)
        depth_scaled = depth
        cv2.imwrite(depth_path, depth_scaled)

        # Append metadata
        self.data.append({
            "color_path": color_path,
            "depth_path": depth_path,
            "pose": pose.tolist()
        })

        self.frame_id += 1

    def save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.data, f, indent=4)

class Camera:
    def __init__(self, H=1080, W=1920, fx=1080, fy=1080):
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.cx = self.W/2
        self.cy = self.H/2
        self.Ks = torch.tensor([[self.fx, 0.0, self.cx],
               [0.0, self.fy, self.cy],
               [0.0,  0.0,  1.0]], dtype=torch.float32).to("cuda")
        self.Ks = self.Ks.unsqueeze(0)  # shape: [1, 3, 3]
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.T = torch.eye(4, dtype=torch.float32).unsqueeze(0).to("cuda")

    def move_camera_quat(self, tx, ty, tz, qx, qy, qz, qw):
        # Convert quaternion to rotation matrix
        r = R.from_quat([qx, qy, qz, qw])
        R_mat = torch.tensor(r.as_matrix(), dtype=torch.float32)
        t_vec = torch.tensor([tx, ty, tz], dtype=torch.float32)

        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R_mat
        T[:3, 3] = (R_mat @ t_vec)
        self.T = T.unsqueeze(0).to("cuda")  # shape: [1, 4, 4]
        
        return

    # def set_camera_viewpoint(self, tx, ty, tz, pos, yaw, pitch):
    #     # Look direction vector
    #     front = np.array([
    #         np.cos(pitch) * np.sin(yaw),
    #         np.sin(pitch),
    #         np.cos(pitch) * np.cos(yaw)
    #     ])
    #     front = front / np.linalg.norm(front)

    #     up = np.array([0.0, 1.0, 0.0])
    #     right = np.cross(up, front)
    #     up = np.cross(front, right)

    #     R = torch.eye(4, dtype=torch.float32)

    #     R[:3, 0] = torch.from_numpy(right).to(dtype=torch.float32)
    #     R[:3, 1] = torch.from_numpy(up).to(dtype=torch.float32)
    #     R[:3, 2] = torch.from_numpy(-front).to(dtype=torch.float32)

    #     T = torch.eye(4, dtype=torch.float32)
    #     T[:3, 3] = torch.from_numpy(-pos).to(dtype=torch.float32)
    #     cam_pose = R @ T
    #     self.T = cam_pose.unsqueeze(0).to("cuda")  # camera-to-world inverse
    #     return

    def set_camera_viewpoint(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):

        # set camera pose

        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        rot = R.from_euler('zyx', [roll, pitch, yaw], degrees=True)
        R_mat = rot.as_matrix()          # 3x3


        camera_pos = np.array([x, y, z])
        t = np.array(camera_pos, dtype=np.float32).reshape(3, 1)

        np_view = np.eye(4, dtype=np.float32)
        np_view[0:3, 0:3] = R_mat         # inverse rotation

        np_view[0:3, 3] = np.squeeze(R_mat @ t)     # inverse translation


        self.T = torch.from_numpy(np_view).to(dtype=torch.float32).unsqueeze(0).to("cuda")

        return


    def rotate_camera(self, roll=0, pitch=0, yaw=0):
        # Get current rotation matrix from T
        R_curr = self.T[0, 0:3, 0:3].cpu().numpy()  # [3, 3]

        # Compute incremental rotation (local frame)
        R_delta = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

        # Apply local-frame rotation: R_new = R_curr @ R_delta
        R_new = R_curr @ R_delta

        # Update the transformation matrix
        np_view = np.eye(4, dtype=np.float32)
        np_view[0:3, 0:3] = R_new
        camera_pos = torch.tensor([self.x, self.y, self.z], dtype=torch.float32)
        camera_pos = np.squeeze(np.array(camera_pos.cpu(), dtype=np.float32).reshape(3, 1))
        np_view[0:3, 3] = camera_pos

        self.T = torch.from_numpy(np_view).to(dtype=torch.float32).unsqueeze(0).to("cuda")


    def move_camera(self, move_vec):

        self.x += move_vec[0]
        self.y += move_vec[1]
        self.z += move_vec[2]

        # Update camera pose
        self.T[0,0,3] = self.x
        self.T[0,1,3] = self.y
        self.T[0,2,3] = self.z

        return


    def pygame_move_camera(self):
        keys = pygame.key.get_pressed()
        
        move_vec=torch.from_numpy(np.zeros(3)).to(dtype=torch.float32).to("cuda")
        R_mat = self.T.squeeze(0)[0:3, 0:3]
        front = -R_mat[:, 2]
        right = R_mat[:, 0]
        up = R_mat[:, 1]

        roll, pitch, yaw = 0, 0, 0
        
        if keys[K_w]:
            move_vec -= 0.1 * front
            self.move_camera(move_vec)
        if keys[K_s]:
            move_vec += 0.1 * front
            self.move_camera(move_vec)
        if keys[K_d]:
            move_vec += 0.1 * right
            self.move_camera(move_vec)
        if keys[K_a]:
            move_vec -= 0.1 * right
            self.move_camera(move_vec)

        if keys[K_i]:
            roll += 1.0
            self.rotate_camera(roll, pitch, yaw)
        if keys[K_k]:
            roll -= 1.0
            self.rotate_camera(roll, pitch, yaw)
        if keys[K_j]:
            pitch -= 1.0
            self.rotate_camera(roll, pitch, yaw)
        if keys[K_l]:
            pitch += 1.0
            self.rotate_camera(roll, pitch, yaw)
        if keys[K_u]:
            yaw -= 1.0
            self.rotate_camera(roll, pitch, yaw)
        if keys[K_o]:
            yaw += 1.0
            self.rotate_camera(roll, pitch, yaw)

        return

class GaussianModel:

    def __init__(self, sh_degree, ply_path=None, optimizer_type="default"):
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

        # Load ply file
        if ply_path is not None:
            self.load_ply(ply_path)

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
        quats = self._rotation  # [N, 4]

        scales = torch.exp(self._scaling)  # [N, 3]
        opacities = torch.sigmoid(self._opacity)  # [N,]

        image_ids = kwargs.pop("image_ids", None)

        colors = torch.cat((self._features_dc, self._features_rest), 1)  # [N, K, 3]
        rasterize_mode = "antialiased"
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


def render_rgbd_from_obj(cam, obj_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh from {obj_path}")

    mesh.compute_vertex_normals()

    renderer = o3d.visualization.rendering.OffscreenRenderer(cam.W, cam.H)
    renderer.scene.set_background([0, 0, 0, 1])

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, material)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    cam_intrinsics = [cam.fx, cam.fy, cam.cx, cam.cy]
    intrinsic.set_intrinsics(cam.W, cam.H, *cam_intrinsics)

    # extrinsic = np.linalg.inv(cam_pose.cpu().numpy()).astype(np.float32)
    # renderer.setup_camera(intrinsic, extrinsic)
    
    extrinsic = np.linalg.inv(cam.T.squeeze(0).cpu().numpy()).astype(np.float64)
    renderer.setup_camera(intrinsic, extrinsic)

    color = renderer.render_to_image()
    depth = renderer.render_to_depth_image(z_in_view_space=True)

    color_np = np.asarray(color)
    depth_np = np.asarray(depth)

    return color_np, depth_np


def rasterize_rgbd(cam, gaussian_model, near_plane=0.001, far_plane=100.0):

    image_ids = torch.tensor([0], dtype=torch.long)  # Shape: [1]
    masks = torch.ones((1, cam.H, cam.W, 4), dtype=torch.bool)  # Shape: [1, 1080, 1920, 4]

    # 3DGS Renderings
    renders, alphas, info = gaussian_model.rasterize_splats(
        camtoworlds=cam.T,
        Ks=cam.Ks,
        width=cam.W,
        height=cam.H,
        sh_degree=3,
        near_plane=near_plane,
        far_plane=far_plane,
        image_ids=image_ids,
        render_mode="RGB+D",
        masks=masks,
    )
    colors, depths = renders[..., 0:3], renders[..., 3:4]


    return colors, depths



def render_xgrids_pose_file(gaussian_model, render_video=False):

    # Initialize Camera
    H = 1080
    W = 1920
    fx = fy = 1080
    cam = Camera(H, W, fx, fy)

    # Load trajectory
    pose_file =  "img_traj.csv" # "panoramicPoses.csv" "img_traj.csv" "poses.csv"

    img_traj_path = f"/root/code/datasets/ARTGarage/xgrids/1/ResultDataArtGarage_sample_2025-07-17-121502_0/ArtGarage_sample_2025-07-17-121502/{pose_file}"
    df = pd.read_csv(img_traj_path, comment="#", sep='\s+',
                 names=["timestamp", "imgname", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

    # df = pd.read_csv(img_traj_path, comment="#", sep='\s+',
    #              names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])


    if render_video:

        # Define output path
        video_path = "/root/code/datasets/ARTGarage/xgrids/rendered_comparison.mp4"

        # Define video writer (assumes 1920x1080 images → adjust if needed)
        frame_h, frame_w = H, W
        output_size = (frame_w * 2, frame_h * 2)  # side-by-side: RGB | Depth
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps=0.5
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, output_size)
    else:
        display_w, display_h = W, H
        tile_w, tile_h = display_w // 2, display_h // 2

    for idx, row in df.iterrows():
        # if idx > 30:
        #     break

        # imgname = row["imgname"][:-4]
        # img_path = f"/root/code/datasets/ARTGarage/xgrids/1/ResultDataArtGarage_sample_2025-07-17-121502_0/ArtGarage_sample_2025-07-17-121502/perspective/images/{imgname}_2.jpg"
        # gt_img = cv2.imread(img_path)
        # gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        # gt_tensor = ToTensor()(gt_img_rgb).permute(1, 2, 0).unsqueeze(0).to("cuda")  # [1, H, W, 3]

        # Set Camera Viewpoint
        cam.set_camera_viewpoint(row.tx, row.ty, row.tz, row.qx, row.qy, row.qz, row.qw)

        colors, depths = rasterize_rgbd(cam, gaussian_model)

        # # Convert to CPU and numpy
        rendered_rgb_3dgs = colors[0].clamp(0, 1).detach().cpu().numpy()  # [H, W, 3]
        rendered_depth_3dgs = depths[0].squeeze(2).detach().cpu().numpy()  # [H, W]

        # === Convert to displayable format ===
        rgb_vis_3dgs = (rendered_rgb_3dgs * 255).astype(np.uint8)
        rgb_vis_3dgs = cv2.cvtColor(rgb_vis_3dgs, cv2.COLOR_RGB2BGR)
        depth_min = rendered_depth_3dgs.min()
        depth_max = rendered_depth_3dgs.max()
        depth_vis_3dgs = normalize_depth(rendered_depth_3dgs, depth_min, depth_max)
        

        # OBJ File Renderings
        obj_file = "/root/code/datasets/ARTGarage/xgrids/4/Gaussian/Mesh_Files/art_garage_sample.obj"
        #obj_file = "/root/code/datasets/ARTGarage/xgrids/4/Mesh_textured/texture/block0.obj"

        rgb_obj, depth_obj = render_rgbd_from_obj(cam, obj_file)
    
        # Ensure OBJ RGB is BGR
        rgb_vis_obj = cv2.cvtColor(rgb_obj, cv2.COLOR_RGB2BGR)
        # Normalize OBJ depth for visualization
        depth_vis_obj = normalize_depth(depth_obj, depth_min, depth_max)


        if render_video:

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
        else:

            rgb1 = cv2.resize(rgb_vis_3dgs, (tile_w, tile_h))
            rgb2 = cv2.resize(rgb_vis_obj, (tile_w, tile_h))
            depth1 = cv2.resize(depth_vis_3dgs, (tile_w, tile_h))
            depth2 = cv2.resize(depth_vis_obj, (tile_w, tile_h))

            depth_3dgs_rescale = cv2.resize(rendered_depth_3dgs, (tile_w, tile_h))
            depth_obj_rescale = cv2.resize(depth_obj, (tile_w, tile_h))

            cv2.putText(rgb1, "GS RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(rgb2, "OBJ RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(depth1, "GS Depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(depth2, "OBJ Depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            top = np.hstack([rgb1, rgb2])
            bottom = np.hstack([depth1, depth2])
            grid = np.vstack([top, bottom])  # final shape: (1080, 1920, 3)

            cv2.namedWindow("Renderings", cv2.WINDOW_NORMAL)
            cv2.imshow("Renderings", grid)
            cv2.resizeWindow("Renderings", display_w, display_h)
            cv2.setMouseCallback("Renderings", quadrant_callback, param=(rgb1, rgb2, depth1, depth2, depth_3dgs_rescale, depth_obj_rescale))


            # # Add windows and mouse callback
            # cv2.namedWindow("GS RGB")
            # cv2.setMouseCallback("GS RGB", on_mouse_click, param=("GS RGB", rgb_vis_3dgs, rendered_depth_3dgs))
            # cv2.imshow("GS RGB", rgb_vis_3dgs)

            # cv2.namedWindow("GS Depth")
            # cv2.setMouseCallback("GS Depth", on_mouse_click, param=("GS Depth", depth_vis_3dgs, rendered_depth_3dgs))
            # cv2.imshow("GS Depth", depth_vis_3dgs)

            # cv2.namedWindow("OBJ RGB")
            # cv2.setMouseCallback("OBJ RGB", on_mouse_click, param=("OBJ RGB", rgb_vis_3dgs, depth_obj))
            # cv2.imshow("OBJ RGB", rgb_vis_3dgs)

            # cv2.namedWindow("OBJ Depth")
            # cv2.setMouseCallback("OBJ Depth", on_mouse_click, param=("OBJ Depth", depth_vis_obj, depth_obj))
            # cv2.imshow("OBJ Depth", depth_vis_obj)

            print("🔍 Click on any window to annotate pixel values. Press any key to move to the next frame.")
            key = cv2.waitKey(0)
            if key in [27, ord('q')]:  # ESC or 'q'
                print("🛑 Exiting...")
                break

    if render_video:

        video_writer.release()
        print(f"✅ Video saved at: {video_path}")
    else:
        cv2.destroyAllWindows()


def vr_walkthrough_opencv(gaussian_model):

    def on_mouse_click1(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            window, img, depth = param  # Unpack color/depth flag
            val = depth[y, x]
            text = f"({x}, {y}): {val:.3f}"
            cv2.putText(img, text, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imshow(window, img)

    def quadrant_callback1(event, x, y, flags, param):
        rgb, depth, rendered_depth_3dgs = param
        h, w, _ = rgb.shape
        if y < h:
            on_mouse_click1(event, x, y, flags, ("Renderings", rgb, rendered_depth_3dgs))
        elif y >= h:
            on_mouse_click1(event, x, y - h, flags, ("Renderings", depth, rendered_depth_3dgs))

        # re-render grid with updated images
        grid = np.vstack([rgb, depth])
        cv2.imshow("Renderings", grid)


    # Initialize Camera
    H = 1080
    W = 1920
    fx = fy = 1080
    cam = Camera(H, W, fx, fy)
    near_plane, far_plane = 0.001, 30.0
    display_w, display_h = W, H
    tile_w, tile_h = display_w, display_h // 2

    # Track position and orientation of the camera over time
    tx, ty, tz, roll, pitch, yaw = 0, 0, 0, 0, 0, 0
    cam.set_camera_viewpoint(tx, ty, tz, roll, pitch, yaw)

    rgb_folder_path = "/root/code/output/gaussian_splatting/xgrids_vr1/color"
    depth_folder_path = "/root/code/output/gaussian_splatting/xgrids_vr1/depth"
    json_path="/root/code/output/gaussian_splatting/xgrids_vr1/poses.json"


    # --- Main loop ---
    cv2.namedWindow("View", cv2.WINDOW_NORMAL)
    print("Click on any window to annotate pixel values. Use keyboard to move around in the environment.")
    while True:
        # === Call the Gaussian Rasterizer ===
        colors, depths = rasterize_rgbd(cam, gaussian_model, near_plane, far_plane)

        # # Convert to CPU and numpy
        rendered_rgb_3dgs = colors[0].clamp(0, 1).detach().cpu().numpy()  # [H, W, 3]
        rendered_depth_3dgs = depths[0].squeeze(2).detach().cpu().numpy()  # [H, W]

        # === Convert to displayable format ===
        rgb_vis_3dgs = (rendered_rgb_3dgs * 255).astype(np.uint8)
        rgb_vis_3dgs = cv2.cvtColor(rgb_vis_3dgs, cv2.COLOR_RGB2BGR)

        depth_min = near_plane #rendered_depth_3dgs.min()
        depth_max = far_plane #rendered_depth_3dgs.max()
        depth_vis_3dgs = normalize_depth(rendered_depth_3dgs, depth_min, depth_max)

        rgb = cv2.resize(rgb_vis_3dgs, (tile_w, tile_h))
        depth = cv2.resize(depth_vis_3dgs, (tile_w, tile_h))

        cv2.putText(rgb, "GS RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(depth, "GS Depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        grid = np.vstack([rgb, depth])

        cv2.namedWindow("Renderings", cv2.WINDOW_NORMAL)
        cv2.imshow("Renderings", grid)
        cv2.resizeWindow("Renderings", display_w, display_h)
        cv2.setMouseCallback("Renderings", quadrant_callback1, param=(rgb, depth, rendered_depth_3dgs))

        pose = cam.T.squeeze(0).detach().cpu().numpy().astype(np.float32)

        #recorder.record(rgb_vis_3dgs, depth_vis_3dgs, pose)
        #cv2.imshow("View", rgb_vis_3dgs)

        
        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break

        move_vec=torch.from_numpy(np.zeros(3)).to(dtype=torch.float32).to("cuda")
        R_mat = cam.T.squeeze(0)[0:3, 0:3]
        front = -R_mat[:, 2]
        right = R_mat[:, 0]
        up = R_mat[:, 1]

        if key == ord('w'):
            move_vec -= 0.1 * front
            cam.move_camera(move_vec)

        if key == ord('s'):
            move_vec += 0.1 * front
            cam.move_camera(move_vec)

        if key == ord('a'):
            move_vec -= 0.1 * right
            cam.move_camera(move_vec)

        if key == ord('d'):
            move_vec += 0.1 * right
            cam.move_camera(move_vec)


        roll, pitch, yaw = 0, 0, 0
        # Rotate
        if key == ord('i'):
            roll += 1.0
            cam.rotate_camera(roll, pitch, yaw)

        if key == ord('k'):
            roll -= 1.0
            cam.rotate_camera(roll, pitch, yaw)            

        if key == ord('j'):
            pitch -= 1.0
            cam.rotate_camera(roll, pitch, yaw)

        if key == ord('l'):
            pitch += 1.0
            cam.rotate_camera(roll, pitch, yaw)            

        if key == ord('u'):
            yaw -= 1.0
            cam.rotate_camera(roll, pitch, yaw)

        if key == ord('o'):
            yaw += 1.0
            cam.rotate_camera(roll, pitch, yaw)

    cv2.destroyAllWindows()

def vr_walkthrough_pygame(gaussian_model):

    # Initialize Camera
    H = 1080
    W = 1920
    fx = fy = 1080
    cam = Camera(H, W, fx, fy)
    near_plane, far_plane = 0.001, 30.0

    # Track position and orientation of the camera over time
    tx, ty, tz, roll, pitch, yaw = 0, 0, 0, 0, 0, 0
    cam.set_camera_viewpoint(tx, ty, tz, roll, pitch, yaw)

    rgb_folder_path = "/root/code/output/gaussian_splatting/xgrids_vr1/color"
    depth_folder_path = "/root/code/output/gaussian_splatting/xgrids_vr1/depth"
    json_path="/root/code/output/gaussian_splatting/xgrids_vr1/poses.json"


    # os.makedirs(rgb_folder_path, exist_ok=True)
    # os.makedirs(depth_folder_path, exist_ok=True)


    pygame.init()
    display_width, display_height = W, H
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Gaussian Splat Viewpoint Control")

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()
    running = True
    recorder = Recorder(rgb_folder_path, depth_folder_path, json_path)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            # elif event.type == pygame.MOUSEMOTION:
            #     handle_mouse_input(event)

        cam.pygame_move_camera()

        # === Call the Gaussian Rasterizer ===
        colors, depths = rasterize_rgbd(cam, gaussian_model, near_plane, far_plane)

        # # Convert to CPU and numpy
        rendered_rgb_3dgs = colors[0].clamp(0, 1).detach().cpu().numpy()  # [H, W, 3]
        rendered_depth_3dgs = depths[0].squeeze(2).detach().cpu().numpy()  # [H, W]

        # === Convert to displayable format ===
        rgb_vis_3dgs = (rendered_rgb_3dgs * 255).astype(np.uint8)
        depth_min = rendered_depth_3dgs.min()
        depth_max = rendered_depth_3dgs.max()
        depth_vis_3dgs = normalize_depth(rendered_depth_3dgs, depth_min, depth_max)


        pose = cam.T.squeeze(0).detach().cpu().numpy().astype(np.float32)

        recorder.record(rgb_vis_3dgs, depth_vis_3dgs, pose)
        # === Display the output image ===
        # Convert the NumPy array to a Pygame surface
        # The swapaxes() is crucial because NumPy arrays and Pygame surfaces have different memory layouts.
        # NumPy is (height, width, channels), Pygame is (width, height, channels).
        image_surface = pygame.surfarray.make_surface(rgb_vis_3dgs.swapaxes(0, 1))

        # Blit the surface to the screen
        screen.blit(image_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    recorder.save_json()
    pygame.quit()
    sys.exit()


def main():
    # parser = argparse.ArgumentParser(description="Load trained Gaussian Splat stored as a ply file and render RGBD images.")
    # parser.add_argument("--render_video", type=bool, default=False, help="Render Video or Images")
    # args = parser.parse_args()

    ply_file_path="/root/code/datasets/ARTGarage/xgrids/4/Gaussian/PLY_Generic_splats_format/point_cloud/iteration_100/point_cloud.ply"
    gaussian_model = GaussianModel(3, ply_file_path)

    # render_xgrids_pose_file(gaussian_model, render_video=True)

    # vr_walkthrough_opencv(gaussian_model)
    vr_walkthrough_pygame(gaussian_model)


if __name__ == "__main__":
    main()
