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
from pathlib import Path
from enum import Enum

import supervision as sv
import pycocotools.mask as mask_util
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import random
import grounding_dino.groundingdino.datasets.transforms as T
from transformers import CLIPProcessor, CLIPModel
import gc
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def conic_to_ellipse(conic, mean, scale=1.0, num_pts=60):
    """
    conic: [A,B,C]
    mean: (u,v) in pixels
    scale: scaling factor for sigma (1=sigma, 2=2sigma, etc.)
    """
    A, B, C = conic
    u, v = mean

    # Precision matrix
    P = np.array([[A, B],
                  [B, C]], dtype=np.float64)

    # Invert to covariance
    cov = np.linalg.inv(P + 1e-6 * np.eye(2))

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Radii = sqrt eigenvalues
    radii = np.sqrt(eigvals) * scale

    # Ellipse points
    t = np.linspace(0, 2*np.pi, num_pts)
    circle = np.stack([np.cos(t) * radii[0], np.sin(t) * radii[1]], axis=0)

    # Rotate
    ellipse = eigvecs @ circle

    # Translate to mean
    ellipse[0] += u
    ellipse[1] += v

    return ellipse.T  # [num_pts, 2]



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


class Record_Mode(Enum):
    PAUSE=0
    RECORD=1
    CONTINUE=2

class Recorder:
    def __init__(self, out_dir=Path("/root/code/output/gaussian_splatting/xgrids_vr/"), record_mode=Record_Mode.PAUSE):

        self.out_dir = out_dir
        self.save_json_flag = False
        self.pose_data = {}
        self.noisy_pose_data = {}
        self.frame_id = 0

        if record_mode == Record_Mode.CONTINUE:
            self.load_previous_state()
        else:
            self.init_recorder()

    def init_recorder(self, suffix=""):
        suffix = f"_{suffix}" if suffix else ""
        self.color_dir = self.out_dir / f"color{suffix}"
        self.depth_dir = self.out_dir / f"depth{suffix}"
        self.seg_dir = self.out_dir / f"seg{suffix}"
        self.norm_depth_dir = self.out_dir / f"norm_depth{suffix}"
        self.screen_capture_dir = self.out_dir / f"screen_capture{suffix}"
        self.json_path = self.out_dir / f"poses{suffix}.json"

        self.pose_data = {}
        self.noisy_pose_data = {}
        self.frame_id = 0
        self.sc_frame_id = 0

        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.seg_dir, exist_ok=True)
        os.makedirs(self.norm_depth_dir, exist_ok=True)
        os.makedirs(self.screen_capture_dir, exist_ok=True)

        print(f"🔄 Intializing recording....")

    def load_previous_state(self):
        """Resume from previous recording if json exists"""
        self.json_path = self.out_dir / "poses.json"
        self.color_dir = self.out_dir / "color"
        self.depth_dir = self.out_dir / "depth"
        self.seg_dir = self.out_dir / "seg"
        self.norm_depth_dir = self.out_dir / "norm_depth"
        self.screen_capture_dir = self.out_dir / "screen_capture"

        # Load JSON if it exists
        if self.json_path.exists():
            with open(self.json_path, "r") as f:
                data = json.load(f)
            self.pose_data = data.get("poses", {})
            self.noisy_pose_data = data.get("noisy_poses", {})

            # Set frame_id to last + 1
            if len(self.pose_data) > 0:
                self.frame_id = max(int(k) for k in self.pose_data.keys()) + 1
            else:
                self.frame_id = 0
        else:
            self.pose_data = {}
            self.noisy_pose_data = {}
            self.frame_id = 0

        self.sc_frame_id = 0
        print(f"🔄 Resuming recording at frame {self.frame_id}")

    def record(self, rgb: np.ndarray, depth: np.ndarray, norm_depth: np.ndarray, pose: np.ndarray, noisy_pose: Optional[np.ndarray] = None, seg: Optional[np.ndarray] = None):
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
        norm_depth_path = os.path.join(self.norm_depth_dir, name)

        # Save images
        cv2.imwrite(color_path, rgb)
        cv2.imwrite(depth_path, depth)
        cv2.imwrite(norm_depth_path, norm_depth)

        if seg is not None:
            seg_path = os.path.join(self.seg_dir, name)
            cv2.imwrite(seg_path, seg)

        # Append metadata
        self.pose_data[self.frame_id]= pose.tolist()
        if noisy_pose is not None:
            self.noisy_pose_data[self.frame_id] = noisy_pose.tolist()

        self.frame_id += 1

    def save_json(self):
        all_data = {
            "poses": self.pose_data,
            "noisy_poses": self.noisy_pose_data
        }
        with open(self.json_path, 'w') as f:
            json.dump(all_data, f, indent=4)

        print(f"✅ saved json file to {self.json_path}")

    def screen_capture(self, rgb: np.ndarray, depth: np.ndarray, norm_depth: np.ndarray, pose: np.ndarray, Ks, noisy_pose: Optional[np.ndarray] = None, seg: Optional[np.ndarray] = None):
        # File names
        name = f"sc_frame_color_{self.sc_frame_id:04d}.png"
        color_path = os.path.join(self.screen_capture_dir, name)
        name = f"sc_frame_depth_{self.sc_frame_id:04d}.png"
        depth_path = os.path.join(self.screen_capture_dir, name)
        name = f"sc_frame_norm_depth_{self.sc_frame_id:04d}.png"
        norm_depth_path = os.path.join(self.screen_capture_dir, name)
        name = f"sc_pose_{self.sc_frame_id:04d}.json"
        json_path = os.path.join(self.screen_capture_dir, name)

        # Save images
        cv2.imwrite(color_path, rgb)
        cv2.imwrite(depth_path, depth)
        cv2.imwrite(norm_depth_path, norm_depth)

        if seg is not None:
            name = f"sc_frame_norm_seg_{self.sc_frame_id:04d}.png"
            seg_path = os.path.join(self.screen_capture_dir, name)
            cv2.imwrite(seg_path, seg)

        print(f"Camera Pose {pose}")

        all_data = {
            "Ks": Ks.cpu().numpy().tolist(),
            "pose": pose.tolist()
        }
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=4)

        print(f"saved json file to {json_path}")

        self.sc_frame_id += 1

    def record_camera_poses(self, camera_poses):
        pose_data = {}
        for i, cam_pose in enumerate(camera_poses):
            pose_data[i] = cam_pose
        all_data = {
            "poses": pose_data
        }
        json_path = self.out_dir / "recorded_camera_poses.json"
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=4)

        print(f"✅ saved json file to {json_path}")

class Camera:
    def __init__(self, H=1080, W=1920, fx=1080, fy=1080, near_plane=0.001, far_plane=100.0):
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.near_plane=near_plane
        self.far_plane=far_plane
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
        self.recorded_poses = []

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

    def set_camera_viewpoint(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

        # Rotation: world→camera or camera→world?
        # For GSplat we need **camera-to-world**
        R_mat = R.from_euler('zyx', [roll, pitch, yaw], degrees=True).as_matrix()

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_mat
        T[:3, 3] = np.array([x, y, z], dtype=np.float32)

        self.T = torch.from_numpy(T).unsqueeze(0).cuda()


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


        # print(f"x: {self.x}, y:{self.y}, z:{self.z}")

        return

    def print_camera_pose(self):
        print(f"x: {self.x}, y: {self.y}, z: {self.z}")
        print(f"roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}")

class GaussianModel:

    def __init__(self, sh_degree, ply_path=None, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._feature_field = torch.empty(0) # Semantic feature field
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
            packed=True,
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

# ------------------------------
# Dataset: samples random pixels from saved feature maps
# ------------------------------

class CLIPFeatureDataset(Dataset):
    def __init__(self, feature_dir, sample_size=100_000):
        self.files = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))
        assert len(self.files) > 0, f"No .pt files found in {feature_dir}"
        self.sample_size = sample_size

        print(f"📁 Found {len(self.files)} feature maps in {feature_dir}")
        print(f"🎯 Sampling total {sample_size:,} random feature vectors")

        self.samples = []
        per_file = max(1, sample_size // len(self.files))

        for f in tqdm(self.files, desc="Loading feature maps"):
            feat = torch.load(f, map_location="cpu")  # [H, W, 512]
            feat = feat.view(-1, 512)
            n = min(per_file, feat.shape[0])
            idx = torch.randperm(feat.shape[0])[:n]
            self.samples.append(feat[idx])

        self.samples = torch.cat(self.samples, dim=0)
        print(f"✅ Loaded {self.samples.shape[0]:,} samples total")

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]


# ------------------------------
# Model: simple fully-connected autoencoder
# ------------------------------

class CLIPAutoencoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

class SplatSegmenter:
    def __init__(self, input_dir=None):

        """
        Hyper parameters
        """
        #self.text = "floor. wall. pillars. plants. door. window. fan. chair. machine. fire extinguisher. shoe rack. traffic cone. chain."
        #self.text = "floor. wall. pillars."
        self.text = "floor. wall. curtain. door. window. fan. chair. table. mouse. keyboard. glass. water dispenser. fire extinguisher. laptop. \
            monitor. phone. cupboard. bag. lock. chessboard. cup. soda. bin. drawer. \
            tv. light. pen. cloth."

        self.class_names = [cls.strip().rstrip('.') for cls in self.text.split('.') if cls.strip()]
        self.clip_embedding_classes = self.class_names + ["others"]

        # Initialize CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.generate_embeddings() # Generate Clip Embeddings for defined object classes
    

        # === Create consistent color map across runs ===
        random.seed(42)  # ensure same colors every run
        self.CLASS_COLOR_MAP = {
            cls: sv.Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for cls in self.class_names
        }


        self.sam2_checkpoint = "/root/code/langsam/checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.GROUNDING_DINO_CONFIG = "/root/code/langsam/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT = "/root/code/langsam/gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.OUTPUT_DIR = None
        self.FEATURE_DIR = None

        # if input_dir is not None:
        #     input_dir = "/root/code/output/xgrids_vr_test"

        self.img_dir = input_dir / "color"    

        if self.img_dir.exists():
            self.OUTPUT_DIR = input_dir / "seg"
            # create output directory
            self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            self.FEATURE_DIR = input_dir / "features"
            self.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        else:
            assert self.img_dir.exists(), f"❌ Directory not found: {self.img_dir}"


        # build SAM2 image predictor
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT,
            device=self.device
        )

    def segment_splat_image_dir(self, record_mode=True):

        """
        Run LangSAM segmentation on all images in a directory.
        """


        # Supported image formats
        extensions = [".jpg", ".jpeg", ".png"]

        # list all images
        for frame_id, img_path in enumerate(sorted(self.img_dir.iterdir())):
            # if frame_id > 20:
            #     break
            if img_path.suffix.lower() not in extensions:
                continue  # skip non-image files
            print(f"🔹 Processing: {img_path}")

            image_source, image = load_image(img_path)

            # Run LangSAM segmentation
            annotated_frame_bgr, feature_map_norm = self.langsam_gaussian_segmenter(image_source, image)
            print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

            if record_mode:
                cv2.imwrite(os.path.join(self.OUTPUT_DIR, img_path.name), annotated_frame_bgr)

                # if 1:
                #     # ✅ Save feature map as .pt (PyTorch tensor)
                #     feature_save_path = self.FEATURE_DIR / (img_path.stem + ".pt")
                #     torch.save(feature_map_norm, feature_save_path)
                #     print(f"💾 Saved feature map: {feature_save_path}")
                # else:
                #     pass

                #     # # ✅ Save feature map as .npy (numpy)
                #     # np.save(self.FEATURE_DIR / (img_path.stem + ".npy"), feature_map_norm.cpu().numpy())
                #     # Clear unused variables and caches

                # ✅ Save AnyLabeling JSON annotation
                self.save_anylabeling_json(
                    img_path=img_path,
                    gdino_labels=self.last_gdino_labels,
                    boxes=self.last_boxes,
                    masks=self.last_masks,
                    scores=self.last_confidences
                )

            del image_source, image, annotated_frame_bgr, feature_map_norm
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    def langsam_gaussian_segmenter(self, image_source, image):

        # environment settings
        # use bfloat16



        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot
        

        # Grounding DINO

        self.sam2_predictor.set_image(image_source)

        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16):
            boxes, gdino_detection_confidences, gdino_detection_labels = predict(
                model=self.grounding_model,
                image=image,
                caption=self.text,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
                device=self.device
            )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # # FIXME: figure how does this influence the G-DINO model
        # torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float32

        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=autocast_dtype):
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        gdino_detection_confidences = gdino_detection_confidences.numpy().tolist()

        # Store for JSON export
        self.last_boxes = input_boxes.tolist()
        self.last_confidences = gdino_detection_confidences
        self.last_gdino_labels = gdino_detection_labels
        self.last_masks = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks

        class_ids = np.array(list(range(len(gdino_detection_labels))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(gdino_detection_labels, gdino_detection_confidences)
        ]
        #print(labels)

        """
        Visualize image with supervision useful API
        """

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        # --- Compute per-detection colors manually ---
        colors = []
        for class_id in detections.class_id:
            cls = gdino_detection_labels[int(class_id)] if class_id < len(gdino_detection_labels) else "unknown"
            color = self.CLASS_COLOR_MAP.get(cls, sv.Color(255, 255, 255))
            colors.append(color.as_bgr())

        # --- Draw boxes and masks ---
        annotated_frame = image_source.copy()

        # Draw bounding boxes and labels
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box)
            color = colors[i]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label_text = labels[i] if i < len(labels) else ""
            cv2.putText(
                annotated_frame, label_text,
                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        # Apply colored masks
        for i, mask in enumerate(detections.mask):
            color = colors[i]
            colored_mask = np.zeros_like(annotated_frame)
            colored_mask[mask] = color
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.4, 0)

        # Generate Feature map

        # Use the precomputed "others" embedding
        others_embedding = self.embeddings["others"]

        # Get the shape of the image
        H, W, _ = image_source.shape
        
        # Initialize the feature map with the "others" embedding
        feature_map = others_embedding.repeat(H * W, axis=0).reshape(H, W, 512)

        # Apply the class-specific embeddings for each mask
        DISPLAY_FLAG = False
        for i, mask in enumerate(detections.mask):
            class_name = gdino_detection_labels[i]

            # GDINO can generate labels that are not in the classes defined (limited testing results show a bounding box generated can have multiple labels)
            # (label generated is just a string with detected class labels appended to each other seperated with spaces)
            # Some more observations - GDINO can produce multiple bounding boxes for the same object (again limited testing)
            # example plants in concrete pots produce bboxes for plants and plants plus the pots, but SAM segments only plants for both bounding boxes
            # Another example is fans, here SAM is making mistakes sometimes
            # Extensive testing required. Similar images are generating different results sometimes..


            if class_name not in self.class_names:
                DISPLAY_FLAG = True
                #print(f"Detected class label {class_name} not in classes defined.... Not processing mask!!!")
                continue
            class_vector = self.embeddings[class_name]
            feature_map[mask > 0] = class_vector
            #print(f"gdino_class_name: {class_name}, feature_map: {feature_map[mask > 0].shape} feature_vec: {class_vector.shape}")
        feature_map = torch.nn.functional.normalize(torch.tensor(feature_map, device = "cuda"), dim=-1)
                
        # # for 512->16 
        # if compress:
        #     feature_map_norm = feature_map_norm @ encoder_decoder.encoder



        # ============================
        # 🟦 ADD VISUAL LEGEND ON RIGHT
        # ============================

        legend_font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        swatch_size = 25
        line_spacing = 10
        text_color = (255, 255, 255)

        # Compute legend dimensions
        legend_entries = list(self.CLASS_COLOR_MAP.items())
        legend_height = (swatch_size + line_spacing) * len(legend_entries) + 40
        legend_width = 300

        # Match image height
        img_h, img_w = annotated_frame.shape[:2]
        legend_img = np.zeros((img_h, legend_width, 3), dtype=np.uint8)

        # Fill legend background (dark gray)
        legend_img[:] = (30, 30, 30)

        # Draw title
        cv2.putText(legend_img, "Legend", (20, 35), legend_font, 0.8, (200, 200, 200), 2)

        # Draw color swatches and labels
        y0 = 70
        for cls, color in legend_entries:
            bgr = color.as_bgr()
            # Draw color box
            cv2.rectangle(legend_img, (20, y0), (20 + swatch_size, y0 + swatch_size), bgr, -1)
            # Write class name
            cv2.putText(legend_img, cls, (60, y0 + swatch_size - 5), legend_font, font_scale, text_color, font_thickness)
            y0 += swatch_size + line_spacing

        # Combine horizontally
        final_display = np.hstack([annotated_frame, legend_img])
        final_display_bgr = cv2.cvtColor(final_display, cv2.COLOR_RGB2BGR)


        # # ============================
        # # 🖼️ Display
        # # ============================
        # if DISPLAY_FLAG:
        #     cv2.namedWindow("grounded_sam2", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("grounded_sam2", 1600, 720)
        #     cv2.imshow("grounded_sam2", final_display_bgr)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return final_display_bgr, feature_map

    def generate_embeddings(self):
        # Compute the "others" embedding only once
        self.embeddings = {}
        for prompt in self.clip_embedding_classes:
            inputs = self.clip_processor(text=prompt, return_tensors="pt", padding=True).to("cuda")
            text_feat = self.clip_model.get_text_features(**inputs)  # Shape: [1, 512] (embedding for "others")
            norm_embedding = torch.nn.functional.normalize(text_feat, p=2, dim=1)
            self.embeddings[prompt] = norm_embedding.detach().cpu().numpy() # Detach the tensor and convert it to a NumPy array    


        for key, embedding in self.embeddings.items():
            print(f"Class: {key}, Embedding Shape: {embedding.shape}")
        return

    def save_anylabeling_json(self, img_path, gdino_labels, boxes, masks, scores):
        """
        Save Grounded SAM2 detections in AnyLabeling-compatible JSON format.
        Ensures all coordinates are ints (Qt crashes otherwise).
        """
        import cv2, json
        from PIL import Image

        # Read actual image dimensions
        img = Image.open(img_path)
        W, H = img.size

        shapes = []

        for i, (label, box, score) in enumerate(zip(gdino_labels, boxes, scores)):
            # --- round/convert coordinates to int ---
            x1, y1, x2, y2 = [int(round(v)) for v in box]

            # rectangle shape
            shapes.append({
                "label": str(label),
                "text": f"{score:.2f}",
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })

            # --- Optional: polygon mask ---
            if masks is not None and len(masks) > i:
                mask = masks[i].astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    polygon = contour.squeeze(1)
                    if len(polygon.shape) != 2 or polygon.shape[0] < 3:
                        continue
                    # ensure integers
                    polygon = [[int(round(float(x))), int(round(float(y)))] for x, y in polygon]
                    shapes.append({
                        "label": str(label),
                        "text": f"{score:.2f}",
                        "points": polygon,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })

        json_data = {
            "version": "0.4.30",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": None,
            "imageHeight": H,
            "imageWidth": W,
            "text": ""
        }

        json_save_path = self.OUTPUT_DIR / f"{img_path.stem}.json"
        with open(json_save_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"💾 Saved AnyLabeling JSON (no floats): {json_save_path}")



class SPLAT_APP:
    def __init__(self, cam, gaussian_model, recorder):

        self.gaussian_model = gaussian_model


        # store a deep copy of original gaussians
        self.original_gaussians = {
            "_xyz": self.gaussian_model._xyz.clone(),
            "_rotation": self.gaussian_model._rotation.clone(),
            "_scaling": self.gaussian_model._scaling.clone(),
            "_opacity": self.gaussian_model._opacity.clone(),
            "_features_dc": self.gaussian_model._features_dc.clone(),
            "_features_rest": self.gaussian_model._features_rest.clone(),
        }

        self.selected_gaussians = {
            "_xyz": None,
            "_rotation": None,
            "_scaling": None,
            "_opacity": None,
            "_features_dc": None,
            "_features_rest": None,
            "gaussian_ids": set(),
        } # Use this for single view operations


        self.object_gaussians = {
            "_xyz": None,
            "_rotation": None,
            "_scaling": None,
            "_opacity": None,
            "_features_dc": None,
            "_features_rest": None,
            "gaussian_ids": set(),
        } # Use this for single view operations


        self.removed_gaussians = {
            "_xyz": None,
            "_rotation": None,
            "_scaling": None,
            "_opacity": None,
            "_features_dc": None,
            "_features_rest": None,
            "gaussian_ids": set(),
        } # Use this for single view operations


        self.background_gaussians = {
            "_xyz": None,
            "_rotation": None,
            "_scaling": None,
            "_opacity": None,
            "_features_dc": None,
            "_features_rest": None,
            "gaussian_ids": set(),
        } # Use this for single view operations

        self.gaussian_visualization_settings = {
            "enabled": False,
            "on_rgb": False,
            "on_depth": False,
            "selected_gaussians": False,
            "object_gaussians": False,
            "removed_gaussians": False,
            "background_gaussians": False,
        }

        self.cam = cam
        self.recorder = recorder
        self.splat_segmenter = SplatSegmenter(recorder.out_dir)
        self.transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        #Define noise parameters for replaying training/recorded trajectories with noise injected

        # for Uniformly Random Noise
        self.trans_noise=0.1
        self.rot_noise_deg=5.0

        # for Normal Noise
        self.std_dev_trans_noise=0.1
        self.std_dev_rot_noise_deg=5.0

    def rasterize_rgbd(self):

        image_ids = torch.tensor([0], dtype=torch.long)  # Shape: [1]
        masks = torch.ones((1, self.cam.H, self.cam.W, 4), dtype=torch.bool)  # Shape: [1, 1080, 1920, 4]

        # 3DGS Renderings
        sh_degree = self.gaussian_model._sh_degree
        renders, alphas, meta = self.gaussian_model.rasterize_splats(
            camtoworlds=self.cam.T,
            Ks=self.cam.Ks,
            width=self.cam.W,
            height=self.cam.H,
            sh_degree=sh_degree,
            near_plane=self.cam.near_plane,
            far_plane=self.cam.far_plane,
            image_ids=image_ids,
            render_mode="RGB+D",
            masks=masks,
        )
        colors, depths = renders[..., 0:3], renders[..., 3:4]


        return colors, depths, alphas, meta

    def rasterize_images(self, visualize_gaussians=False):

        # === Call the Gaussian Rasterizer ===
        colors, depths, alphas, meta = self.rasterize_rgbd()

        # # Convert to CPU and numpy
        rendered_rgb_3dgs = colors[0].clamp(0, 1).detach().cpu().numpy()  # [H, W, 3]
        rendered_depth_3dgs = depths[0].squeeze(2).detach().cpu().numpy()  # [H, W]

        # === Convert to displayable format ===
        rgb_vis_3dgs = (rendered_rgb_3dgs * 255).astype(np.uint8)

        depth_min = rendered_depth_3dgs.min()
        depth_max = rendered_depth_3dgs.max()
        depth_vis_3dgs = normalize_depth(rendered_depth_3dgs, depth_min, depth_max)


        rgb_vis_3dgs_bgr = cv2.cvtColor(rgb_vis_3dgs, cv2.COLOR_RGB2BGR)

        if visualize_gaussians:
            rgb_vis_3dgs_bgr = self.overlay_gaussians_on_image(rgb_vis_3dgs_bgr, meta)
            rgb_vis_3dgs = self.overlay_gaussians_on_image(rgb_vis_3dgs, meta)
            rendered_depth_3dgs = self.overlay_gaussians_on_image(rendered_depth_3dgs, meta)
            depth_vis_3dgs = self.overlay_gaussians_on_image(depth_vis_3dgs, meta)


        return rgb_vis_3dgs_bgr, rgb_vis_3dgs, rendered_depth_3dgs, depth_vis_3dgs


    def overlay_gaussians_on_image(self, img, meta, ellipse_std=2.0):
        """
        Draws Gaussian outlines on a grayscale or RGB image.
        Works for depth, RGB, or depth_vis.
        """
        if img is None:
            return img

        means2d = meta["means2d"].detach().cpu().numpy()
        conics  = meta["conics"].detach().cpu().numpy()
        ids     = meta["gaussian_ids"].detach().cpu().numpy()

        # Visualization sets
        selected_gids     = self.selected_gaussians["gaussian_ids"] if self.gaussian_visualization_settings["selected_gaussians"] else []
        removed_gids      = self.removed_gaussians["gaussian_ids"] if self.gaussian_visualization_settings["removed_gaussians"] else []
        object_gids   = self.object_gaussians["gaussian_ids"] if self.gaussian_visualization_settings["object_gaussians"] else []
        bg_gids   = self.background_gaussians["gaussian_ids"] if self.gaussian_visualization_settings["background_gaussians"] else []


        # Convert gray → 3-channel
        if len(img.shape) == 2:  
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(len(ids)):
            gid = ids[i]
            if gid < 0:
                continue

            # Assign colors
            if gid in selected_gids:
                color = (0, 0, 128)
            elif gid in removed_gids:
                color = (0, 255, 0)
            elif gid in object_gids:
                color = (255, 165, 0)
            elif gid in bg_gids:
                color = (255, 0, 255)
            else:
                continue

            # 2D gaussian center + ellipse
            u, v = means2d[i]
            A, B, C = conics[i]
            ellipse_pts = self.sample_conic_ellipse((A, B, C), (u, v), k=ellipse_std)
            pts = ellipse_pts.reshape(-1, 1, 2).astype(np.int32)

            # Draw
            #cv2.polylines(img, [pts], True, color, 1)
            cv2.circle(img, (int(u), int(v)), 2, color, -1)

        return img

    def conic_to_ellipse(conic, mean, scale=1.0, num_pts=60):
        """
        conic: [A,B,C]
        mean: (u,v) in pixels
        scale: scaling factor for sigma (1=sigma, 2=2sigma, etc.)
        """
        A, B, C = conic
        u, v = mean

        # Precision matrix
        P = np.array([[A, B],
                      [B, C]], dtype=np.float64)

        # Invert to covariance
        cov = np.linalg.inv(P + 1e-6 * np.eye(2))

        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Radii = sqrt eigenvalues
        radii = np.sqrt(eigvals) * scale

        # Ellipse points
        t = np.linspace(0, 2*np.pi, num_pts)
        circle = np.stack([np.cos(t) * radii[0], np.sin(t) * radii[1]], axis=0)

        # Rotate
        ellipse = eigvecs @ circle

        # Translate to mean
        ellipse[0] += u
        ellipse[1] += v

        return ellipse.T  # [num_pts, 2]



    def overlay_gaussians(self, base_image, meta, obj_mask=None, bg_mask=None):
        """
        base_image: HxWx3 RGB, uint8
        meta: gsplat metadata from rasterizer
        obj_mask, bg_mask: uint8 masks (optional)
        """

        img = base_image.copy()
        means2d = meta["means2d"].detach().cpu().numpy()   # [M,2]
        conics  = meta["conics"].detach().cpu().numpy()    # [M,3]
        ids     = meta["gaussian_ids"].detach().cpu().numpy()  # [M]

        H, W = img.shape[:2]

        for i in range(len(ids)):
            u, v = means2d[i]
            if u < 0 or v < 0 or u >= W or v >= H:
                continue

            mean = (u, v)
            conic = conics[i]

            # Sample 1-sigma and 2-sigma
            e1 = conic_to_ellipse(conic, mean, scale=1.0)
            e2 = conic_to_ellipse(conic, mean, scale=2.0)

            # Determine color
            if obj_mask is not None and obj_mask[int(v), int(u)] > 0:
                col = (0,255,0)     # green
            elif bg_mask is not None and bg_mask[int(v), int(u)] > 0:
                col = (0,0,255)     # red
            else:
                col = (255,255,255) # neutral white

            # Draw ellipses
            e1 = e1.astype(np.int32)
            e2 = e2.astype(np.int32)
            cv2.polylines(img, [e1], isClosed=True, color=col, thickness=1)
            cv2.polylines(img, [e2], isClosed=True, color=col, thickness=1)

            # Draw small mean point
            cv2.circle(img, (int(u),int(v)), 1, col, -1)

        return img


    def restore_original_gaussians(self):

        self.gaussian_model._xyz         = self.original_gaussians["_xyz"].clone()
        self.gaussian_model._rotation    = self.original_gaussians["_rotation"].clone()
        self.gaussian_model._scaling     = self.original_gaussians["_scaling"].clone()
        self.gaussian_model._opacity     = self.original_gaussians["_opacity"].clone()
        self.gaussian_model._features_dc = self.original_gaussians["_features_dc"].clone()
        self.gaussian_model._features_rest = self.original_gaussians["_features_rest"].clone()

    def store_selected_gaussians(self, selected_gaussian_ids):
        obj_gids_list = sorted(list(selected_gaussian_ids))

        self.selected_gaussians = {
            "_xyz": self.gaussian_model._xyz[obj_gids_list].clone(),
            "_rotation": self.gaussian_model._rotation[obj_gids_list].clone(),
            "_scaling": self.gaussian_model._scaling[obj_gids_list].clone(),
            "_opacity": self.gaussian_model._opacity[obj_gids_list].clone(),
            "_features_dc": self.gaussian_model._features_dc[obj_gids_list].clone(),
            "_features_rest": self.gaussian_model._features_rest[obj_gids_list].clone(),
            "gaussian_ids": selected_gaussian_ids,
        } # Use this for single view operations


    def update_object_gaussians(self, selected_gaussian_ids):

        self.object_gaussians["gaussian_ids"].update(selected_gaussian_ids)
        obj_gids = self.object_gaussians["gaussian_ids"]
        obj_gids_list = sorted(list(obj_gids))

        self.object_gaussians = {
            "_xyz": self.gaussian_model._xyz[obj_gids_list].clone(),
            "_rotation": self.gaussian_model._rotation[obj_gids_list].clone(),
            "_scaling": self.gaussian_model._scaling[obj_gids_list].clone(),
            "_opacity": self.gaussian_model._opacity[obj_gids_list].clone(),
            "_features_dc": self.gaussian_model._features_dc[obj_gids_list].clone(),
            "_features_rest": self.gaussian_model._features_rest[obj_gids_list].clone(),
            "gaussian_ids": obj_gids,
        } # Use this for single view operations

        self.update_background_gaussians(selected_gaussian_ids)


    def reset_object_gaussians(self):

        self.object_gaussians = {
            "_xyz": None,
            "_rotation": None,
            "_scaling": None,
            "_opacity": None,
            "_features_dc": None,
            "_features_rest": None,
            "gaussian_ids": set()
        } # Use this for single view operations


    def update_background_gaussians(self, selected_gaussian_ids):
        N = self.gaussian_model._xyz.shape[0]
        all_gids = set(range(N))

        self.object_gaussians["gaussian_ids"].update(selected_gaussian_ids)
        obj_gids = self.object_gaussians["gaussian_ids"]
        bg_gids = all_gids - obj_gids
        bg_gids_list = sorted(list(bg_gids))

        self.background_gaussians = {
            "_xyz": self.gaussian_model._xyz[bg_gids_list].clone(),
            "_rotation": self.gaussian_model._rotation[bg_gids_list].clone(),
            "_scaling": self.gaussian_model._scaling[bg_gids_list].clone(),
            "_opacity": self.gaussian_model._opacity[bg_gids_list].clone(),
            "_features_dc": self.gaussian_model._features_dc[bg_gids_list].clone(),
            "_features_rest": self.gaussian_model._features_rest[bg_gids_list].clone(),
            "gaussian_ids": bg_gids,
        } # Use this for single view operations



    def make_object_gaussians_invisible(self):
        selected_gaussian_ids = np.array(list(self.object_gaussians["gaussian_ids"]), dtype=np.int32)
        with torch.no_grad():
            self.gaussian_model._opacity[selected_gaussian_ids] = -10.0


    def remove_object_gaussians(self):
        """
        Remove the selected Gaussians
        and update the gaussian_model to contain only the remaining ones.
        """

        if not hasattr(self, "selected_gaussians"):
            print("⚠️ No gaussians selected. Run select_object_interactively() first.")
            return

        selected_gaussian_ids = np.array(list(self.object_gaussians["gaussian_ids"]), dtype=np.int32)

        remove_idx = torch.from_numpy(selected_gaussian_ids).to("cuda")
        N = self.gaussian_model._xyz.shape[0]

        mask = torch.ones(N, device="cuda", dtype=torch.bool)
        mask[remove_idx] = False   # keep everything else

        print(f"🗑️ Removing {remove_idx.numel()} gaussians, keeping {mask.sum().item()}")

        # Update all gaussian attributes
        with torch.no_grad():
            self.gaussian_model._xyz        = self.gaussian_model._xyz[mask]
            self.gaussian_model._rotation   = self.gaussian_model._rotation[mask]
            self.gaussian_model._scaling    = self.gaussian_model._scaling[mask]
            self.gaussian_model._opacity    = self.gaussian_model._opacity[mask]
            self.gaussian_model._features_dc   = self.gaussian_model._features_dc[mask]
            self.gaussian_model._features_rest = self.gaussian_model._features_rest[mask]


    def shift_object_gaussians(self, 
                                 translation=(0,0,0), 
                                 rotation_deg=(0,0,0)):
        """
        Apply a rigid transform to selected Gaussians: rotation + translation.

        Args:
            translation: (dx, dy, dz) in world coordinates
            rotation_deg: (yaw, pitch, roll) in degrees
        """
        if not hasattr(self, "selected_gaussians"):
            print("⚠️ No gaussians selected. Run select_object_interactively() first.")
            return

        selected_gaussian_ids = np.array(list(self.object_gaussians["gaussian_ids"]), dtype=np.int32)


        sel = torch.from_numpy(selected_gaussian_ids).to("cuda")

        # ---- Extract dx, dy, dz ----
        dx, dy, dz = translation
        t = torch.tensor([dx, dy, dz], device="cuda", dtype=torch.float32)

        # ---- Build rotation matrix ----
        yaw, pitch, roll = rotation_deg
        R_mat = R.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_matrix()
        R_torch = torch.tensor(R_mat, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            # -------- 1. Update positions --------
            xyz = self.gaussian_model._xyz[sel]
            xyz_new = (xyz @ R_torch.T) + t
            self.gaussian_model._xyz[sel] = xyz_new

            # -------- 2. Update rotations --------
            # original rotations are stored as quaternions in _rotation
            q = self.gaussian_model._rotation[sel]  # [N, 4] (w,x,y,z)
            q_rot = R.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_quat()
            q_rot_t = torch.tensor(q_rot, device="cuda", dtype=torch.float32)
            
            # quaternion multiply: q_new = q_rot * q_old
            # (apply the new rotation in front)
            def quat_mul(q1, q2):
                # q1, q2: [N,4]
                w1,x1,y1,z1 = q1.unbind(-1)
                w2,x2,y2,z2 = q2.unbind(-1)
                return torch.stack([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2
                ], dim=-1)

            q_rot_rep = q_rot_t.expand_as(q)
            self.gaussian_model._rotation[sel] = quat_mul(q_rot_rep, q)

        print(f"✅ Shifted {sel.numel()} gaussians by translation={translation}, rotation={rotation_deg}")



    # def remove_gaussian(self, gid):
    #     with torch.no_grad():
    #         self.gaussian_model._opacity[gid] = -10.0

    # def remove_dominant_gaussian(self):
    #     with torch.no_grad():
    #         self.gaussian_model._opacity[self.selected_gaussians[0]] = -10.0


    # def split_and_highlight_gaussian(self, gid):
    #     with torch.no_grad():
    #         self.gaussian_model._features_dc[gid] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    #         self.gaussian_model._opacity[gid] = 5.0

    # def highlight_gaussian(self, gid):
    #     with torch.no_grad():
    #         self.gaussian_model._features_dc[gid] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    #         self.gaussian_model._opacity[gid] = 5.0


    # def highlight_selected_gaussians(self):
    #     with torch.no_grad():
    #         self.gaussian_model._features_dc[self.selected_gaussians] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    #         self.gaussian_model._opacity[self.selected_gaussians] = 5.0

    # def highlight_dominant_gaussian(self):
    #     with torch.no_grad():
    #         self.gaussian_model._features_dc[self.selected_gaussians[0]] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    #         self.gaussian_model._opacity[self.selected_gaussians[0]] = 5.0


    def select_object_interactively(self):

        # Render current view
        rgb_vis_3dgs_bgr, rgb_vis_3dgs, rendered_depth_3dgs, depth_vis_3dgs = self.rasterize_images()

        # Run interactive SAM2 on the RGB image
        mask_overlay, pos_points, neg_points = self.interactive_sam2_segmentation(rgb_vis_3dgs_bgr)

        if mask_overlay is None:
            print("⚠️ No mask selected (user quit).")
            return None

        # Make sure mask is H x W uint8
        if mask_overlay.ndim == 3:
            mask_overlay = mask_overlay.squeeze()
        mask_overlay = (mask_overlay > 0).astype(np.uint8)

        dist = cv2.distanceTransform(mask_overlay.astype(np.uint8), cv2.DIST_L2, 5)
        norm_dist = dist / dist.max()     # normalize 0–1

        if 1:
            mask_clean = norm_dist * mask_overlay
        elif 0:
            mask_clean = (dist > 2).astype(np.uint8)     # keep pixels that are >2 px inside
        else:
            mask_clean = mask_overlay

        # Map 2D mask → Gaussians
        gaussian_idx = self.mask_to_gaussian_indices(mask_clean, min_grad=1e-5)

        # Store for later editing
        self.selected_gaussian_ids = gaussian_idx
        print(f"🎯 Stored {len(gaussian_idx)} selected Gaussians in self.selected_gaussians")

        return gaussian_idx

    def select_object_interactively_v2(self):

        """
        Interactive SAM2 segmentation using positive/negative clicks.

        Controls:
            Left-click  = positive prompt
            Right-click = negative prompt
            ENTER       = finalize and return mask
            r           = reset all points
            q           = quit (returns None)
        """


        device="cuda"
        dtype=torch.bfloat16

        sam_window_name = "Interactive SAM2 Segmentation"
        cv2.namedWindow(sam_window_name, cv2.WINDOW_NORMAL)
        edited_gaussian_window_name = "Interactive SAM2 Segmentation"
        cv2.namedWindow(edited_gaussian_window_name, cv2.WINDOW_NORMAL)

        image_rgb, _, _, _ = self.rasterize_images()

        H, W, _ = image_rgb.shape

        img_disp = image_rgb.copy()
        pos_points = []
        neg_points = []
        mask_overlay = None

        # ---- wrapper so callback has correct scope ----
        def callback(event, x, y, flags, param):
            nonlocal pos_points, neg_points, mask_overlay, img_disp

            if event == cv2.EVENT_LBUTTONDOWN:
                pos_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg_points.append((x, y))
            else:
                return

            # --- run SAM with updated points ---
            mask_overlay = self.run_sam2_points(
                image_rgb=image_rgb,
                pos_points=pos_points,
                neg_points=neg_points,
                device=device,
                dtype=dtype
            )

            # --- redraw ---
            img_disp = self.draw_interaction(
                image_rgb=image_rgb,
                pos_points=pos_points,
                neg_points=neg_points,
                mask_overlay=mask_overlay
            )

        cv2.setMouseCallback(sam_window_name, callback)

        # === main loop ===
        while True:
            cv2.imshow(sam_window_name, img_disp)
            key = cv2.waitKey(20)

            if key == 13:  # ENTER
                cv2.destroyWindow(sam_window_name)

                if mask_overlay is None:
                    print("⚠️ No mask selected (user quit).")
                else:
                    # Make sure mask is H x W uint8
                    if mask_overlay.ndim == 3:
                        mask_overlay = mask_overlay.squeeze()
                    obj_mask_overlay = (mask_overlay > 0).astype(np.uint8)


                    obj_selected_gaussians = self.select_gaussians_via_2d_conics(obj_mask_overlay)


                    #obj_selected_gaussians = self.select_gaussians_via_2d_conics_vectorized(obj_mask_overlay)


                    # obj_selected_gaussians = self.select_gaussians_via_2d_conics_gpu(obj_mask_overlay)
                    



                    # bg_mask_overlay = (1 - obj_mask_overlay).astype(np.uint8)

                    # # Map 2D mask → Gaussians
                    # obj_selected_gaussians = self.mask_to_gaussian_indices(obj_mask_overlay, min_grad=1e-5)
                    # bg_selected_gaussians = self.mask_to_gaussian_indices(bg_mask_overlay, min_grad=1e-5)
                    # self.selected_gaussians = bg_selected_gaussians


                    self.selected_gaussian_ids = obj_selected_gaussians

                    

                    # Edit Gaussians
                    #self.highlight_selected_gaussians()
                    #self.highlight_dominant_gaussian()
                    self.remove_selected_gaussians()
                    #self.remove_dominant_gaussian()

                    # for i, gid in enumerate(self.selected_gaussians):
                    #     self.remove_gaussian(gid)
                    #     # Rasterize and Display to user
                    #     rgb_bgr, _, _, _ = self.rasterize_images()
                    #     print(f"Removed {i}-th Gaussian and rendering")
                    #     cv2.imshow(edited_gaussian_window_name, rgb_bgr)
                    #     key = cv2.waitKey(0)

                    # Rasterize and Display to user
                    rgb_bgr, _, _, _ = self.rasterize_images()
                    cv2.putText(rgb_bgr, f"Found {len(self.selected_gaussians)} Gaussians contributing to that pixel.", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow(edited_gaussian_window_name, rgb_bgr)
                    key = cv2.waitKey(0)
                    cv2.destroyWindow(edited_gaussian_window_name)
            elif key == ord('r'):

                # Restore Original Gaussians
                self.restore_original_gaussians()
                image_rgb, _, _, _ = self.rasterize_images()

                # Reset
                pos_points = []
                neg_points = []
                mask_overlay = None
                img_disp = image_rgb.copy()
                cv2.imshow(edited_gaussian_window_name, image_rgb)

            elif key == ord('q'):
                cv2.destroyWindow(sam_window_name)
                return

    def select_object_gaussians_multiview(self):

        # Iterate over Recorded Camera Poses
        selected_gaussian_ids = set()
        self.reset_object_gaussians()
        self.restore_original_gaussians()

        self.gaussian_visualization_settings = {
            "enabled": True,
            "on_rgb": True,
            "on_depth": True,
            "selected_gaussians": False,
            "object_gaussians": True,
            "removed_gaussians": False,
            "background_gaussians": False,
        }
        edited_gaussian_window_name1 = "selected_gaussian_window_rgb"
        edited_gaussian_window_name2 = "selected_gaussian_window_depth"

        if len(self.cam.recorded_poses) > 0:
            for T in self.cam.recorded_poses:
                self.cam.T = T
                obj_selected_gaussians, obj_mask_overlay = self.select_object_gaussians_interactive_sam(prune_method_id=1)
                self.store_selected_gaussians(obj_selected_gaussians)
                self.update_object_gaussians(obj_selected_gaussians)
                #self.make_selected_gaussians_invisible()

                img_bgr, img_rgb, depth, depth_norm = self.rasterize_images(visualize_gaussians=True)
                #cv2.putText(img_bgr, f"Found {len(obj_selected_gaussians)} Gaussians contributing to the object.", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow(edited_gaussian_window_name1, img_bgr)
                #cv2.imshow(edited_gaussian_window_name2, depth)
                key = cv2.waitKey(0)

                cv2.destroyWindow(edited_gaussian_window_name1)
                #cv2.destroyWindow(edited_gaussian_window_name2)

                self.restore_original_gaussians()

            self.make_object_gaussians_invisible()

            self.gaussian_visualization_settings = {
                "enabled": True,
                "on_rgb": True,
                "on_depth": True,
                "selected_gaussians": False,
                "object_gaussians": True,
                "removed_gaussians": False,
                "background_gaussians": False,
            }

            # Rasterize and Display to user from all views
            edited_gaussian_window_name = "Gaussian Edited Image"
            for T in self.cam.recorded_poses:
                self.cam.T = T
                img_bgr, img_rgb, depth, depth_norm = self.rasterize_images(visualize_gaussians=True)
                #cv2.putText(img_bgr, f"Found {len(obj_selected_gaussians)} Gaussians contributing to the object.", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow(edited_gaussian_window_name1, img_bgr)
                #cv2.imshow(edited_gaussian_window_name2, depth)
                key = cv2.waitKey(0)
                cv2.destroyWindow(edited_gaussian_window_name1)
                #cv2.destroyWindow(edited_gaussian_window_name2)

        return


    def select_object_gaussians_interactive_sam(self, prune_method_id=0):

        device = "cuda"
        dtype = torch.bfloat16
        #show_gaussians = False

        sam_window_name = "Interactive SAM2 Segmentation"
        cv2.namedWindow(sam_window_name, cv2.WINDOW_NORMAL)

        image_rgb, _, _, _ = self.rasterize_images()

        H, W, _ = image_rgb.shape

        img_disp = image_rgb.copy()
        pos_points = []
        neg_points = []
        mask_overlay = None
        obj_mask_overlay = None   # ✅ Prevent unbound variable
        obj_selected_gaussians = None

        # ------------------ callback ------------------
        def callback(event, x, y, flags, param):
            nonlocal pos_points, neg_points, mask_overlay, img_disp, obj_mask_overlay

            if event == cv2.EVENT_LBUTTONDOWN:
                pos_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg_points.append((x, y))
            else:
                return

            # --- run SAM ---
            mask_overlay = self.run_sam2_points(
                image_rgb=image_rgb,
                pos_points=pos_points,
                neg_points=neg_points,
                device=device,
                dtype=dtype
            )

            # --- sanitize mask ---
            if mask_overlay is not None:
                if mask_overlay.ndim == 3:
                    mask_overlay = mask_overlay.squeeze()
                obj_mask_overlay = (mask_overlay > 0).astype(np.uint8)
            else:
                obj_mask_overlay = None

            # --- redraw interface ---
            img_disp = self.draw_interaction(
                image_rgb=image_rgb,
                pos_points=pos_points,
                neg_points=neg_points,
                mask_overlay=mask_overlay
            )

        cv2.setMouseCallback(sam_window_name, callback)

        # ------------------ main loop ------------------
        while True:

            cv2.imshow(sam_window_name, img_disp)
            key = cv2.waitKey(20)

            # finalize
            if key == 13:  # ENTER
                if obj_mask_overlay is None:
                    print("⚠️ Please click to segment before pressing ENTER.")
                    continue

                # Select Gaussians
                obj_selected_gaussians = self.mask_to_gaussian_indices(
                    obj_mask_overlay, min_grad=1e-5, prune_method_id=prune_method_id
                )

                cv2.putText(
                    img_disp,
                    f"Found {len(obj_selected_gaussians)} Gaussians.",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(sam_window_name, img_disp)
                cv2.waitKey(0)
                cv2.destroyWindow(sam_window_name)

                return obj_selected_gaussians, obj_mask_overlay

            # reset
            elif key == ord('r'):
                pos_points = []
                neg_points = []
                mask_overlay = None
                obj_mask_overlay = None
                img_disp = image_rgb.copy()

            # quit
            elif key == ord('q'):
                cv2.destroyWindow(sam_window_name)
                return obj_selected_gaussians, obj_mask_overlay

    def transfer_saved_features_to_splat(self, input_dir):
        """
        Args:
            input_dir: Directory from which to parse images, poses and features
        Returns:
            gaussian_features: [N, D] per-Gaussian semantic embedding
        """


        # === Read poses from JSON ===
        json_path = input_dir / "poses.json"

        device = "cuda"
        embed_dim = 32

        # === Read poses from JSON ===
        with open(json_path, "r") as f:
            data = json.load(f)
        pose_data = data["poses"]

        gaussian_features = torch.zeros((self.gaussian_model._xyz.shape[0], embed_dim), device=device)
        gaussian_denoms = torch.ones((self.gaussian_model._xyz.shape[0]), device=device) * 1e-12
        del self.splat_segmenter, data
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # === Iterate over all frames ===
        for frame_id_str, pose_list in pose_data.items():
            frame_id = int(frame_id_str)
            print(f"processing frame id: {frame_id}...")
            if frame_id > 20:
                break
            feat_path = input_dir / "features" / "latent_32d" / f"frame_{frame_id:04d}.pt"
            if not feat_path.exists():
                print(f"⚠️ Missing feature map: {feat_path}")
                continue

            feats = torch.load(feat_path).to(device)  # [H, W, D]
            H, W, _ = feats.shape

            # === Convert JSON pose to torch tensor ===
            pose = torch.tensor(pose_list, dtype=torch.float32, device=device)  # [4,4]
            viewmat = torch.linalg.inv(pose.unsqueeze(0))  # camera-to-world → world-to-camera

            # === Rasterize for gradient flow ===
            colors_feats = torch.zeros((self.gaussian_model._xyz.shape[0], embed_dim), device=device, dtype=torch.float32)
            colors_feats.requires_grad_(True)
            colors_feats_0 = torch.zeros(self.gaussian_model._xyz.shape[0], 3, device=device, dtype=torch.float32)
            colors_feats_0.requires_grad_(True)

            # 1️⃣ Numerator rasterization

            output_for_grad, _, _ = rasterization(
                means=self.gaussian_model._xyz,
                quats=self.gaussian_model._rotation,
                scales=torch.exp(self.gaussian_model._scaling),
                opacities=torch.sigmoid(self.gaussian_model._opacity),
                colors=colors_feats.unsqueeze(0),  # ✅ [1, N, D] (C=1, N, K)
                viewmats=viewmat,
                Ks=self.cam.Ks,
                width=W,
                height=H,
            )

            target_feat = (output_for_grad[0].to(torch.float16) * feats.to(torch.float16)).sum()
            target_feat.backward()

            gaussian_features += colors_feats.grad.clone()
            colors_feats.grad.zero_()

            # 2️⃣ Denominator rasterization
            output_denom, _, _ = rasterization(
                means=self.gaussian_model._xyz,
                quats=self.gaussian_model._rotation,
                scales=torch.exp(self.gaussian_model._scaling),
                opacities=torch.sigmoid(self.gaussian_model._opacity),
                colors=colors_feats_0.unsqueeze(0),  # ✅ [1, N, 1]
                viewmats=viewmat,
                Ks=self.cam.Ks,
                width=W,
                height=H,
            )

            target_denom = (output_denom[0]).sum()   # just sum all contributions
            target_denom.backward()

            gaussian_denoms += colors_feats_0.grad[:, 0]
            colors_feats_0.grad.zero_()
            print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


            # Clear unused variables and caches
            del viewmat, feats, output_for_grad, output_denom, target_feat, target_denom
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gaussian_features = gaussian_features / gaussian_denoms[..., None]
        gaussian_features = torch.nn.functional.normalize(gaussian_features, dim=-1)
        gaussian_features[torch.isnan(gaussian_features)] = 0
        print("✅ Computed per-Gaussian semantic features.")
        
        self.gaussian_model._feature_field = gaussian_features

        return 


    def mask_to_gaussian_indices(self, obj_mask_np: np.ndarray, min_grad: float = 1e-5, prune_method_id = 0):
        """
        Given a 2D binary mask over the rendered image, find which Gaussians
        contributed to the pixels inside the mask.

        Args:
            obj_mask_np: (H, W) uint8 or bool numpy mask (1 inside, 0 outside)
            min_grad: threshold on gradient magnitude to keep a Gaussian

        Returns:
            selected_idx: np.ndarray of shape [M] with Gaussian indices
        """
        device = "cuda"
        assert obj_mask_np.ndim == 2, f"Expected 2D mask, got shape {obj_mask_np.shape}"

        H, W = obj_mask_np.shape
        N = self.gaussian_model._xyz.shape[0]

        # Dummy per-Gaussian scalar "color" that we will differentiate w.r.t.
        colors_feats = torch.zeros((N, 1), device=device, dtype=torch.float32, requires_grad=True)

        # Camera/view
        viewmats = torch.linalg.inv(self.cam.T.detach())  # [1, 4, 4]

        # Rasterize with dummy scalar colors
        colors, alphas, meta = rasterization(
            means=self.gaussian_model._xyz.detach(),
            quats=self.gaussian_model._rotation.detach(),
            scales=torch.exp(self.gaussian_model._scaling.detach()),
            opacities=torch.sigmoid(self.gaussian_model._opacity.detach()),
            colors=colors_feats.view(1, N, 1),          # [1, N, 1]
            viewmats=viewmats,                          # [1, 4, 4]
            Ks=self.cam.Ks.detach(),                    # [1, 3, 3]
            width=W,
            height=H,
        )  # colors: [1, H, W, 1]

        # Convert mask to torch and apply to rendered scalar

        obj_mask_t = torch.from_numpy(obj_mask_np.astype(np.float32)).to(device=device)
        obj_mask_t = obj_mask_t.view(H, W, 1)  # [H, W, 1]

        # Generate background mask
        bg_mask_np = (1 - obj_mask_np).astype(np.uint8)
        bg_mask_t = torch.from_numpy(bg_mask_np.astype(np.float32)).to(device=device)
        bg_mask_t = bg_mask_t.view(H, W, 1)  # [H, W, 1]


        # Sum over masked pixels → scalar
        target = (colors[0] * obj_mask_t).sum()
        target.backward()

        # Gradient w.r.t. per-Gaussian scalar tells you which Gaussians contributed
        grad = colors_feats.grad.view(-1).abs()  # [N]

        # Select those above some threshold
        selected = (grad > min_grad).nonzero(as_tuple=False).view(-1)

        object_gaussians = set(selected.cpu().numpy())
        print(f"✅ mask_to_gaussian_indices: selected {len(object_gaussians)} / {N} gaussians")

        # Approach 1: Reject Gaussians with means outside the object mask
        ids = meta["gaussian_ids"].detach().cpu().numpy()   # [M]
        conics = meta["conics"].detach().cpu().numpy()  # [M,3]
        means2d = meta["means2d"].detach().cpu().numpy()  # [M,2]
        global_to_local = {gid: i for i, gid in enumerate(ids)}

        selected_list = list(object_gaussians)

        filtered = []

        # Apply your filters to remove some of the influential Gaussians

        if prune_method_id == 1:

            

            # Approach 1: Reject Gaussians with means outside the object mask

            for gid in selected_list:
                if gid not in global_to_local:
                    # gaussian is not visible in this view → skip
                    continue
                
                local_idx = global_to_local[gid]

                u, v = means2d[local_idx]   


                u, v = int(u), int(v)
                if 0 <= u < W and 0 <= v < H:
                    if obj_mask_np[v, u] == 1:
                        filtered.append(gid)

            object_gaussians = set(filtered)


        elif prune_method_id == 2:

            # Approach 2: Reject Gaussians whose projected 2D ellipse does NOT overlap the mask

            def fast_conic_inside_mask(mean, conic, mask):
                u0, v0 = mean
                A, B, C = conic
                P = np.array([[A, B],[B, C]])
                Sigma = np.linalg.inv(P + 1e-6*np.eye(2))
                sx = int(max(1, 3*np.sqrt(Sigma[0,0])))
                sy = int(max(1, 3*np.sqrt(Sigma[1,1])))
                u0, v0 = int(u0), int(v0)
                vmin, vmax = max(0, v0-sy), min(mask.shape[0], v0+sy)
                umin, umax = max(0, u0-sx), min(mask.shape[1], u0+sx)
                return mask[vmin:vmax, umin:umax].max() > 0


            for gid in selected_list:
                if gid not in global_to_local:
                    # gaussian is not visible in this view → skip
                    continue
                
                local_idx = global_to_local[gid]

                u, v = means2d[local_idx]
                if fast_conic_inside_mask((u,v), conics[local_idx], obj_mask_np):
                    filtered.append(gid)

            object_gaussians = set(filtered)

        elif prune_method_id == 3:

            # Approach 3: PRUNE lOW OPACITY GAUSSIANS
            op = torch.sigmoid(self.gaussian_model._opacity).detach().cpu().numpy()

            filtered = [idx for idx in object_gaussians if op[idx] > 0.05]

            object_gaussians = set(filtered)


        return object_gaussians

    def select_gaussians_via_2d_conics(self, object_mask, k=2.0):
        """
        object_mask: HxW uint8 array from SAM
        k: ellipse cov scaling (2.0 = 95% Gaussian influence)
        """

        # Generate background mask
        background_mask = (1 - object_mask).astype(np.uint8)

        # --- Render to get gsplat meta information ---
        rgb, rgb_linear, depth, depth_vis = self.rasterize_images()
        meta = self._last_meta   # store meta from rasterizer

        means2d = meta["means2d"].detach().cpu().numpy()          # [M,2]
        conics  = meta["conics"].detach().cpu().numpy()           # [M,3]
        ids     = meta["gaussian_ids"].detach().cpu().numpy()     # [M]

        H, W = object_mask.shape

        object_gaussians = []
        background_gaussians = []

        print("🔍 Checking 2D conic overlaps for all visible Gaussians...")

        for i in range(len(ids)):
            u, v = means2d[i]

            # Skip off-screen or invalid projections
            if not (0 <= u < W and 0 <= v < H):
                continue

            ellipse_pts = self.sample_conic_ellipse(conics[i], (u, v), k=k)

            in_obj = self.ellipse_overlaps_mask(ellipse_pts, object_mask)
            in_bg  = self.ellipse_overlaps_mask(ellipse_pts, background_mask)

            gauss_id = ids[i]

            if in_obj:
                object_gaussians.append(gauss_id)

            if in_bg:
                background_gaussians.append(gauss_id)

        object_gaussians = set(object_gaussians)
        background_gaussians = set(background_gaussians)

        # Remove ONLY pure object Gaussians
        #removable = np.array(list(object_gaussians - background_gaussians), dtype=np.int32)

        removable = object_gaussians - background_gaussians

        print(f"🎯 Object Gaussians: {len(object_gaussians)}")
        print(f"🎯 Background Gaussians: {len(background_gaussians)}")
        print(f"🔥 Removable Gaussians: {len(removable)}")

        return removable


    # def select_gaussians_via_2d_conics_vectorized(self, object_mask, k=2.0, ds=8):
    #     """
    #     Fast vectorized version of Gaussian selection using downsampled conics.
    #     - object_mask: HxW uint8 mask
    #     - k: scaling factor for ellipse radius (2.0 ≈ 95% Gaussian energy)
    #     - ds: downsample factor (8–16 recommended)
    #     """

    #     # Background mask
    #     background_mask = (1 - object_mask).astype(np.uint8)

    #     # Render to get meta
    #     _, _, _, _ = self.rasterize_images()
    #     meta = self._last_meta

    #     means2d = meta["means2d"].detach().cpu().numpy()    # (M,2)
    #     conics  = meta["conics"].detach().cpu().numpy()     # (M,3)
    #     ids     = meta["gaussian_ids"].detach().cpu().numpy()  # (M,)

    #     H, W = object_mask.shape

    #     # -----------------------------
    #     # 1. Downsample masks (huge speed-up)
    #     # -----------------------------
    #     Hs, Ws = H // ds, W // ds
    #     obj_small = cv2.resize(object_mask, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
    #     bg_small  = 1 - obj_small

    #     # -----------------------------
    #     # 2. Downsample Gaussian centers
    #     # -----------------------------
    #     u = (means2d[:, 0] / ds).astype(np.int32)
    #     v = (means2d[:, 1] / ds).astype(np.int32)

    #     # clamp out of bounds
    #     u = np.clip(u, 0, Ws - 1)
    #     v = np.clip(v, 0, Hs - 1)

    #     # -----------------------------
    #     # 3. Convert conics → approximate ellipse radii
    #     # Conic = (A, B, C)
    #     # We ignore B (xy term) for a very fast approximation
    #     # ellipse radii ~ k / sqrt([A,C])
    #     # -----------------------------
    #     A = conics[:, 0]
    #     C = conics[:, 2]

    #     # avoid division by zero
    #     A = np.maximum(A, 1e-6)
    #     C = np.maximum(C, 1e-6)

    #     rx = k / np.sqrt(A) / ds   # x-radius in small image
    #     ry = k / np.sqrt(C) / ds   # y-radius in small image

    #     rx = rx.astype(np.int32)
    #     ry = ry.astype(np.int32)

    #     # -----------------------------
    #     # 4. Compute bounding boxes in downsampled space
    #     # -----------------------------
    #     umin = np.clip(u - rx, 0, Ws - 1)
    #     umax = np.clip(u + rx, 0, Ws - 1)
    #     vmin = np.clip(v - ry, 0, Hs - 1)
    #     vmax = np.clip(v + ry, 0, Hs - 1)

    #     # -----------------------------
    #     # 5. FAST overlap check using bounding box max test
    #     # Instead of checking all pixels, we check:
    #     #    If ANY pixel in the bounding box is object → object_gaussian
    #     #    If ANY pixel in the bounding box is background → background_gaussian
    #     # -----------------------------

    #     # Extract region using boolean masks
    #     obj_hits = (obj_small[vmin, umin] |
    #                 obj_small[vmin, umax] |
    #                 obj_small[vmax, umin] |
    #                 obj_small[vmax, umax])

    #     bg_hits  = (bg_small[vmin, umin] |
    #                 bg_small[vmin, umax] |
    #                 bg_small[vmax, umin] |
    #                 bg_small[vmax, umax])

    #     # -----------------------------
    #     # 6. Final classification
    #     # -----------------------------
    #     object_gaussians     = ids[obj_hits]
    #     background_gaussians = ids[bg_hits]

    #     removable = np.setdiff1d(object_gaussians, background_gaussians)

    #     print(f"⚡ Visible Gaussians: {len(ids)}")
    #     print(f"🎯 Object hits (fast): {len(object_gaussians)}")
    #     print(f"🎯 Background hits (fast): {len(background_gaussians)}")
    #     print(f"🔥 Removable Gaussians (fast): {len(removable)}")

    #     return removable

    # def select_gaussians_via_2d_conics_gpu(self, object_mask, k=2.0, ds=4):
    #     """
    #     Ultra-fast GPU ellipse test.
    #     object_mask : [H,W] uint8 numpy array from SAM
    #     k           : conic scaling factor (2.0 recommended)
    #     ds          : mask downsample factor (4–8 recommended)
    #     """

    #     device = "cuda"

    #     # ---------------------------------------
    #     # 1. Get rasterizer metadata
    #     # ---------------------------------------
    #     _, _, _, _ = self.rasterize_images()
    #     meta = self._last_meta

    #     means2d = meta["means2d"].detach().to(device)        # [M,2]
    #     conics  = meta["conics"].detach().to(device)         # [M,3] (A,B,C)
    #     ids     = meta["gaussian_ids"].detach().cpu().numpy()

    #     M = means2d.shape[0]

    #     H, W = object_mask.shape

    #     # ---------------------------------------
    #     # 2. Downsample object mask
    #     # ---------------------------------------
    #     obj_small = cv2.resize(object_mask, (W//ds, H//ds), interpolation=cv2.INTER_NEAREST)
    #     bg_small  = 1 - obj_small

    #     Hs, Ws = obj_small.shape

    #     # torch masks
    #     obj_t = torch.from_numpy(obj_small).to(device=device, dtype=torch.float32)
    #     bg_t  = torch.from_numpy(bg_small).to(device=device, dtype=torch.float32)

    #     # ---------------------------------------
    #     # 3. Downsample Gaussian centers
    #     # ---------------------------------------
    #     u = (means2d[:,0] / ds)
    #     v = (means2d[:,1] / ds)

    #     # clamp
    #     u = torch.clamp(u, 0, Ws-1)
    #     v = torch.clamp(v, 0, Hs-1)

    #     # ---------------------------------------
    #     # 4. Create coordinate grid for mask
    #     # ---------------------------------------
    #     ys = torch.arange(0, Hs, device=device).float()
    #     xs = torch.arange(0, Ws, device=device).float()
    #     Y, X = torch.meshgrid(ys, xs, indexing='ij')

    #     # Flatten grid for vectorized compute
    #     Xf = X.reshape(-1)        # [P]
    #     Yf = Y.reshape(-1)        # [P]
    #     P  = Xf.shape[0]

    #     # ---------------------------------------
    #     # 5. For each Gaussian, compute conic inequality over all pixels:
    #     #
    #     #   A (x-u)^2 + B (x-u)(y-v) + C (y-v)^2 <= k^2
    #     #
    #     # Fully vectorized over (M Gaussians × P pixels)
    #     # ---------------------------------------

    #     # Expand dims for broadcast
    #     ux = u[:,None]      # [M,1]
    #     vy = v[:,None]      # [M,1]

    #     dx = (Xf[None,:] - ux)      # [M,P]
    #     dy = (Yf[None,:] - vy)      # [M,P]

    #     A = conics[:,0][:,None]     # [M,1]
    #     B = conics[:,1][:,None]
    #     C = conics[:,2][:,None]

    #     conic_val = A*dx*dx + B*dx*dy + C*dy*dy    # [M,P]

    #     inside = (conic_val <= k*k)    # [M,P] boolean tensor

    #     # ---------------------------------------
    #     # 6. Compute overlap with mask
    #     # ---------------------------------------
    #     mask_flat_obj = obj_t.reshape(-1) > 0
    #     mask_flat_bg  = bg_t.reshape(-1) > 0

    #     # For each Gaussian:
    #     #    does ellipse intersect object? background?
    #     in_obj = torch.any(inside[:, mask_flat_obj], dim=1)   # [M]
    #     in_bg  = torch.any(inside[:, mask_flat_bg],  dim=1)   # [M]

    #     # ---------------------------------------
    #     # 7. Final classification
    #     # ---------------------------------------
    #     object_gaussians     = ids[in_obj.cpu().numpy()]
    #     background_gaussians = ids[in_bg.cpu().numpy()]

    #     removable = np.setdiff1d(object_gaussians, background_gaussians)

    #     print(f"⚡ GPU Gaussians visible: {M}")
    #     print(f"🎯 Intersect object: {len(object_gaussians)}")
    #     print(f"🎯 Intersect background: {len(background_gaussians)}")
    #     print(f"🔥 Removable (GPU-accurate): {len(removable)}")

    #     return removable


    def ellipse_overlaps_mask(self, ellipse_pts, mask):
        """
        ellipse_pts: Nx2 float coordinates
        mask: HxW uint8 mask
        """
        H, W = mask.shape

        pts = np.round(ellipse_pts).astype(int)

        valid = (pts[:, 0] >= 0) & (pts[:, 0] < W) & \
                (pts[:, 1] >= 0) & (pts[:, 1] < H)
        pts = pts[valid]

        if len(pts) == 0:
            return False

        return np.any(mask[pts[:, 1], pts[:, 0]] > 0)


    def sample_conic_ellipse(self, conic, center, num_samples=40, k=2.0):
        """
        conic: [A, B, C] defining ellipse A dx^2 + B dx dy + C dy^2 = 1
        center: (u, v)
        Returns Nx2 array of (x, y) sample points along ellipse boundary.
        """

        A, B, C = conic
        # Conic matrix (inverse covariance)
        Q = np.array([[A, B/2],
                      [B/2, C]])

        # Covariance = inverse of Q
        cov2d = np.linalg.inv(Q)

        # Eigen decomposition
        vals, vecs = np.linalg.eigh(cov2d)

        # Radii along principal axes
        rx = k * np.sqrt(vals[0])
        ry = k * np.sqrt(vals[1])

        # Parametric ellipse
        theta = np.linspace(0, 2*np.pi, num_samples)
        circle = np.stack([rx*np.cos(theta), ry*np.sin(theta)], axis=1)  # Nx2

        # Rotate by eigenvectors
        ellipse = circle @ vecs.T

        # Shift to pixel center
        ellipse[:, 0] += center[0]
        ellipse[:, 1] += center[1]

        return ellipse


    def debug_render_original_and_edited(self):
        """
        Renders:
            (1) Original gaussian splat
            (2) Edited gaussian splat
        Displays them side by side for visual comparison.
        """

        # ---- Render edited scene (current model) ----
        rgb_edit, _, _, _ = self.rasterize_images()

        # ---- Temporarily restore ORIGINAL gaussians ----
        with torch.no_grad():
            save_xyz        = self.gaussian_model._xyz.clone()
            save_rot        = self.gaussian_model._rotation.clone()
            save_scaling    = self.gaussian_model._scaling.clone()
            save_opacity    = self.gaussian_model._opacity.clone()
            save_fdc        = self.gaussian_model._features_dc.clone()
            save_frest      = self.gaussian_model._features_rest.clone()

            self.gaussian_model._xyz         = self.original_gaussians["_xyz"].clone()
            self.gaussian_model._rotation    = self.original_gaussians["_rotation"].clone()
            self.gaussian_model._scaling     = self.original_gaussians["_scaling"].clone()
            self.gaussian_model._opacity     = self.original_gaussians["_opacity"].clone()
            self.gaussian_model._features_dc = self.original_gaussians["_features_dc"].clone()
            self.gaussian_model._features_rest = self.original_gaussians["_features_rest"].clone()

        # ---- Render original scene ----
        rgb_orig, _, _, _ = self.rasterize_images()

        # ---- Restore the edited gaussians ----
        with torch.no_grad():
            self.gaussian_model._xyz         = save_xyz
            self.gaussian_model._rotation    = save_rot
            self.gaussian_model._scaling     = save_scaling
            self.gaussian_model._opacity     = save_opacity
            self.gaussian_model._features_dc = save_fdc
            self.gaussian_model._features_rest = save_frest

        # ---- Combine for display ----
        combined = np.hstack([rgb_orig, rgb_edit])

        cv2.namedWindow("Original (left)  |  Edited (right)", cv2.WINDOW_NORMAL)
        cv2.imshow("Original (left)  |  Edited (right)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("✅ Rendered original vs edited comparison.")


    def interactive_sam2_segmentation(self, image_rgb, device="cuda", dtype=torch.bfloat16):
        """
        Interactive SAM2 segmentation using positive/negative clicks.

        Controls:
            Left-click  = positive prompt
            Right-click = negative prompt
            ENTER       = finalize and return mask
            r           = reset all points
            q           = quit (returns None)
        """

        window_name = "Interactive SAM2 Segmentation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        H, W, _ = image_rgb.shape

        img_disp = image_rgb.copy()
        pos_points = []
        neg_points = []
        mask_overlay = None

        # ---- wrapper so callback has correct scope ----
        def callback(event, x, y, flags, param):
            nonlocal pos_points, neg_points, mask_overlay, img_disp

            if event == cv2.EVENT_LBUTTONDOWN:
                pos_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg_points.append((x, y))
            else:
                return

            # --- run SAM with updated points ---
            mask_overlay = self.run_sam2_points(
                image_rgb=image_rgb,
                pos_points=pos_points,
                neg_points=neg_points,
                device=device,
                dtype=dtype
            )

            # --- redraw ---
            img_disp = self.draw_interaction(
                image_rgb=image_rgb,
                pos_points=pos_points,
                neg_points=neg_points,
                mask_overlay=mask_overlay
            )

        cv2.setMouseCallback(window_name, callback)

        # === main loop ===
        while True:
            cv2.imshow(window_name, img_disp)
            key = cv2.waitKey(20)

            if key == 13:  # ENTER
                cv2.destroyWindow(window_name)
                return mask_overlay, pos_points, neg_points

            elif key == ord('r'):
                pos_points = []
                neg_points = []
                mask_overlay = None
                img_disp = image_rgb.copy()

            elif key == ord('q'):
                cv2.destroyWindow(window_name)
                return None, None, None


    # ---------------------------------------------------------------
    # Helper: Run SAM2 with point prompts
    # ---------------------------------------------------------------
    def run_sam2_points(self,
        image_rgb,
        pos_points,
        neg_points,
        device="cuda",
        dtype=torch.bfloat16
    ):

        self.splat_segmenter.sam2_predictor.set_image(image_rgb)

        point_coords = []
        point_labels = []

        if pos_points:
            point_coords.extend(pos_points)
            point_labels.extend([1] * len(pos_points))

        if neg_points:
            point_coords.extend(neg_points)
            point_labels.extend([0] * len(neg_points))

        if len(point_coords) == 0:
            return None

        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            masks, _, _ = self.splat_segmenter.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False  # single best mask
            )

        m0 = masks[0]
        if isinstance(m0, torch.Tensor):
            mask = m0.detach().cpu().numpy().astype(np.uint8)
        else:
            mask = m0.astype(np.uint8)

        return mask


    # ---------------------------------------------------------------
    # Helper: Draw clicks + mask overlay
    # ---------------------------------------------------------------
    def draw_interaction(self, image_rgb, pos_points, neg_points, mask_overlay):
        overlay = image_rgb.copy()

        # draw positive points (green)
        for x, y in pos_points:
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)

        # draw negative points (red)
        for x, y in neg_points:
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)

        # apply mask
        if mask_overlay is not None:
            masked = np.zeros_like(overlay)
            masked[mask_overlay > 0] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 1.0, masked, 0.4, 0)

            masked = np.zeros_like(overlay)
            masked[mask_overlay <= 0] = (255, 0, 0)
            overlay = cv2.addWeighted(overlay, 1.0, masked, 0.4, 0)

        return overlay

    def select_pixel_interactively(self):
        """
        Opens a window where the user clicks on a pixel.
        Returns (x, y) pixel coordinates.
        """
        # Render the scene
        rgb_bgr, rgb, rendered_depth, depth_vis = self.rasterize_images()
        self._last_render = rgb_bgr  # store for use in callback

        window_name = "click_pixel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, rgb_bgr)

        # Mouse callback
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"🖱️  Pixel selected: ({x}, {y})")
                gaussians = self.select_gaussians_at_pixel(x, y)
                # Store them for editing
                self.selected_gaussians = gaussians
                #self.highlight_gaussians()
                #self.highlight_dominant_gaussian()
                self.remove_gaussians()
                #self.remove_dominant_gaussian()
                rgb_bgr, rgb, rendered_depth, depth_vis = self.rasterize_images()
                cv2.putText(rgb_bgr, f"Found {len(gaussians)} Gaussians contributing to that pixel.", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow(window_name, rgb_bgr)

        cv2.setMouseCallback(window_name, mouse_callback)

        # === main loop ===
        while True:
            key = cv2.waitKey(20)

            if key == 13:  # ENTER
                cv2.destroyWindow(window_name)
                return

            elif key == ord('q'):
                cv2.destroyWindow(window_name)
                return
            else:
                continue

        return


    def select_gaussians_at_pixel(self, x, y, min_grad=1e-6):
        """
        Runs gradient-based Gaussian selection at a single pixel (x,y).
        Returns Gaussian indices.
        """
        H, W = self._last_render.shape[:2]

        # 1-pixel mask
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y, x] = 1

        print("🎯 Running mask_to_gaussian_indices on single pixel...")
        selected = self.mask_to_gaussian_indices(mask, min_grad=min_grad)

        print(f"✔ Found {len(selected)} Gaussians contributing to that pixel.")
        return selected



    def vr_walkthrough_pygame(self, record_mode):

        pygame.init()
        pygame.mouse.set_visible(True)
        display_width, display_height = self.cam.W, self.cam.H
        screen = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption("Gaussian Splat Viewpoint Control")

        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

        clock = pygame.time.Clock()
        if record_mode in [Record_Mode.RECORD, Record_Mode.CONTINUE]:
            record_mode = Record_Mode.PAUSE
            self.recorder.save_json_flag = True

        running = True
        show_help_menu = False
        screen_capture = False
        mouse_click_event = False
        clicked_pixel = None
        splat_edit = False
        execute_gaussian_edit = False
        grab_cam_pose = False
        pos_points = []
        neg_points = []
        device="cuda"
        dtype=torch.bfloat16

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:

                        # DUMP poses to recorded_camera_poses.json
                        self.recorder.record_camera_poses(self.cam.recorded_poses)

                        running = False
                    elif event.key == pygame.K_SPACE:
                        if record_mode == Record_Mode.PAUSE:
                            record_mode = Record_Mode.RECORD
                            self.recorder.save_json_flag = True
                        elif record_mode in [Record_Mode.RECORD, Record_Mode.CONTINUE]:
                            record_mode = Record_Mode.PAUSE
                    elif event.key == pygame.K_h:
                        show_help_menu = not show_help_menu
                    elif event.key == pygame.K_c:
                        screen_capture = True
                    elif event.key == pygame.K_r:
                        # Restore Original Gaussians
                        self.restore_original_gaussians()
                    elif event.key == pygame.K_m:
                        splat_edit = not splat_edit
                        pos_points = []
                        neg_points = []
                        print(f"🖱️  Splat Edit Mode Toggled. Mode: {splat_edit}")
                    elif event.key == pygame.K_g:
                        grab_cam_pose = True
                    elif event.key in (pygame.K_KP_ENTER, pygame.K_RETURN):
                        execute_gaussian_edit = True

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_click_event = True
                    x, y = event.pos
                    print(f"🖱️  Pixel selected: ({x}, {y})")
                    clicked_pixel = (x, y)
                    if splat_edit:
                        if event.button == 1: # Left Click
                            pos_points.append(clicked_pixel)
                        elif event.button == 3: #Right click
                            neg_points.append(clicked_pixel)
                        print(f"🖱️  Pose Points: {pos_points}, Negative Points: {neg_points}")


                # elif event.type == pygame.MOUSEMOTION:
                #     handle_mouse_input(event)



            self.cam.pygame_move_camera()

            pose = self.cam.T.squeeze(0).detach().cpu().numpy().astype(np.float32)
            # self.cam.print_camera_pose() # If you want to know where you are in the world

            if grab_cam_pose:
                self.cam.recorded_poses.append(self.cam.T)
                grab_cam_pose = False

            rgb_vis_3dgs_bgr, rgb_vis_3dgs, rendered_depth_3dgs, depth_vis_3dgs = self.rasterize_images()

            if record_mode in [Record_Mode.RECORD, Record_Mode.CONTINUE]:
                self.recorder.record(rgb=rgb_vis_3dgs_bgr, depth=rendered_depth_3dgs, norm_depth=depth_vis_3dgs, pose=pose)

            if screen_capture:
                self.recorder.screen_capture(rgb=rgb_vis_3dgs_bgr, depth=rendered_depth_3dgs, norm_depth=depth_vis_3dgs, pose=pose, Ks=self.cam.Ks)
                screen_capture = False


            # === Display the output image ===
            # Convert the NumPy array to a Pygame surface
            # The swapaxes() is crucial because NumPy arrays and Pygame surfaces have different memory layouts.
            # NumPy is (height, width, channels), Pygame is (width, height, channels).

            rgb_game_img = rgb_vis_3dgs.copy()
            if splat_edit:
                # --- run SAM with updated points ---
                mask_overlay = self.run_sam2_points(
                    image_rgb=rgb_game_img,
                    pos_points=pos_points,
                    neg_points=neg_points,
                    device=device,
                    dtype=dtype
                )

                # --- redraw ---
                rgb_game_img = self.draw_interaction(
                    image_rgb=rgb_game_img,
                    pos_points=pos_points,
                    neg_points=neg_points,
                    mask_overlay=mask_overlay
                )

                if execute_gaussian_edit:
                    print("Executing Splat Edit....")
                    splat_edit = False
                    pos_points = []
                    neg_points = []
                    print(f"🖱️  Splat Edit Mode Toggled. Mode: {splat_edit}")                    


                    # Make sure mask is H x W uint8
                    if mask_overlay.ndim == 3:
                        mask_overlay = mask_overlay.squeeze()
                    obj_mask_overlay = (mask_overlay > 0).astype(np.uint8)
                    # bg_mask_overlay = (1 - obj_mask_overlay).astype(np.uint8)


                    # obj_selected_gaussians = self.select_gaussians_via_2d_conics(obj_mask_overlay)



                    # Map 2D mask → Gaussians
                    obj_selected_gaussians = self.mask_to_gaussian_indices(obj_mask_overlay, min_grad=1e-5)
                    self.selected_gaussians = obj_selected_gaussians

                    # bg_selected_gaussians = self.mask_to_gaussian_indices(bg_mask_overlay, min_grad=1e-5)
                    # self.selected_gaussians = bg_selected_gaussians

                    # Edit Gaussians
                    #self.highlight_selected_gaussians()
                    #self.highlight_dominant_gaussian()
                    self.remove_selected_gaussians()
                    #self.remove_dominant_gaussian()

                    execute_splat_edit = False


            cv2.putText(rgb_game_img, f"Record Mode: {record_mode}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if show_help_menu:
                cv2.putText(rgb_game_img, "Help Menu - Keyboard Shortcuts", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(rgb_game_img, "C - For Screenshot of Current Camera Render", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "WA - For Translation Front/Back Camera Control", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "SD - For Translation Left/Right Camera Control", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "QE - For Translation Up/Down Camera Control", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "IK - YAW Camera Control", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "JL - Pitch Camera Control", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "UO - Roll Camera Control", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "Space Bar - Toggle Recorder", (100, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "M - Edit Gaussian Splat Mode", (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "G - Record Camera Pose", (100, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            if mouse_click_event:
                cv2.circle(rgb_game_img, clicked_pixel, 4, (255, 0, 0), -1)
                mouse_click_event = False


            image_surface = pygame.surfarray.make_surface(rgb_game_img.swapaxes(0, 1))

            # Blit the surface to the screen
            mx, my = pygame.mouse.get_pos()
            pygame.draw.circle(image_surface, (255, 127, 255), (mx, my), 4)
            screen.blit(image_surface, (0, 0))


            pygame.display.flip()
            clock.tick(60)

        if self.recorder.save_json_flag:
            self.recorder.save_json()

        pygame.quit()

    def perturb_pose_normal(self, pose: np.ndarray):
        """
        Args:
            pose: (4,4) numpy array, camera-to-world
            trans_noise: std dev of Gaussian translation noise
            rot_noise_deg: std dev of rotation noise (degrees)
        """
        # Translation noise
        noisy_t = pose[:3, 3] + np.random.normal(0, self.std_dev_trans_noise, 3)

        # Rotation noise
        R_mat = pose[:3, :3]
        noise_rot = R.from_euler('xyz', np.random.normal(0, self.std_dev_rot_noise_deg, 3), degrees=True).as_matrix()
        noisy_R = noise_rot @ R_mat

        noisy_pose = np.eye(4, dtype=np.float32)
        noisy_pose[:3, :3] = noisy_R
        noisy_pose[:3, 3] = noisy_t
        return noisy_pose

    def perturb_pose_uniform(self, pose: np.ndarray):
        """
        Args:
            pose: (4,4) numpy array, camera-to-world
            trans_noise: Absolute maximum noise for uniformly random translation noise
            rot_noise_deg: Absolute maximum noise for uniformly random rotation noise (degrees)
        """
        # Translation noise
        noisy_t = pose[:3, 3] + np.random.uniform(-self.trans_noise, self.trans_noise, 3)

        # Rotation noise
        R_mat = pose[:3, :3]
        noise_rot = R.from_euler('xyz', np.random.uniform(-self.rot_noise_deg, self.rot_noise_deg, 3), degrees=True).as_matrix()
        noisy_R = noise_rot @ R_mat

        noisy_pose = np.eye(4, dtype=np.float32)
        noisy_pose[:3, :3] = noisy_R
        noisy_pose[:3, 3] = noisy_t
        return noisy_pose


    def replay_with_noise(self, pose_file=None, suffix="noisy"):

        # Assuming one vr walk is already done
        # with open(self.recorder.json_path, 'r') as f:
        #     traj = json.load(f)["poses"]

        if pose_file:
            # Load trajectory from a pose file
            traj = self.load_poses(pose_file=pose_file, file_type="csv")
            self.recorder.init_recorder(suffix=f"{suffix}_training")
        else:
            # Load JSON
            if not self.recorder.pose_data:
                self.recorder.init_recorder() # Get the paths
                # Fetch original poses
                print(self.recorder.json_path)
                if self.recorder.json_path.exists():
                    with open(self.recorder.json_path, "r") as f:
                        data = json.load(f)

            # Load pose from trajectory recorded in vr walkthrough
            self.recorder.init_recorder(suffix=suffix)

            traj = data.get("poses", {})


        for frame_id, pose in traj.items():
            pose = np.array(pose, dtype=np.float32)

            # Perturb pose
            noisy_pose = self.perturb_pose_uniform(pose)

            self.cam.T = torch.from_numpy(noisy_pose).unsqueeze(0).cuda()

            colors, depths, alphas, meta = self.rasterize_rgbd()

            # # Convert to CPU and numpy
            rendered_rgb_3dgs = colors[0].clamp(0, 1).detach().cpu().numpy()  # [H, W, 3]
            rendered_depth_3dgs = depths[0].squeeze(2).detach().cpu().numpy()  # [H, W]

            # === Convert to displayable format ===
            rgb_vis_3dgs = (rendered_rgb_3dgs * 255).astype(np.uint8)
            rgb_vis_3dgs_bgr = cv2.cvtColor(rgb_vis_3dgs, cv2.COLOR_RGB2BGR)
            depth_min = rendered_depth_3dgs.min()
            depth_max = rendered_depth_3dgs.max()
            depth_vis_3dgs = normalize_depth(rendered_depth_3dgs, depth_min, depth_max)


            self.recorder.record(rgb_vis_3dgs_bgr, rendered_depth_3dgs, depth_vis_3dgs, pose, noisy_pose)

        self.recorder.save_json()
        print(f"✅ Replay with noise saved to {self.recorder.out_dir.name}")

    def load_poses(self, pose_file: str, file_type="csv"):
        """
        Args:
            pose_file: path to poses
            file_type: 'csv' or 'json'
        Returns:
            dict[int, np.ndarray] mapping frame_id -> 4x4 pose
        """
        poses = {}
        if file_type == "json":
            with open(pose_file, "r") as f:
                data = json.load(f)
            for i, mat in data.items():
                poses[int(i)] = np.array(mat, dtype=np.float32)

        elif file_type == "csv":
            df = pd.read_csv(pose_file, comment="#", sep='\s+',
                             names=["timestamp", "imgname", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
            for idx, row in df.iterrows():
                r = R.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = r
                T[:3, 3] = [row.tx, row.ty, row.tz]
                poses[idx] = T
        else:
            raise ValueError("Unsupported file type")
        return poses


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

def init_cam():

    # For feature database training
    # # Initialize Camera
    # H = 1080
    # W = 1920
    # fx = fy = 1080
    # near_plane, far_plane = 0.001, 100.0
    # cam = Camera(H, W, fx, fy, near_plane, far_plane)


    # For visual inspection and testing
    # Initialize Camera
    H = 720
    W = 1280
    fx = fy = 912
    near_plane, far_plane = 0.001, 100.0
    cam = Camera(H, W, fx, fy, near_plane, far_plane)
    

    # Track position and orientation of the camera over time
    tx, ty, tz, roll, pitch, yaw = 0, 0, 0, 0, 0, 0

    # # Use this initialization for the VR app (Ground floor origin near stairs)
    # tx, ty, tz = 2, -1, 1
    # roll, pitch, yaw = 0, 180, -90

    # # Use this initialization for the VR app (first floor origin near stairs)
    # tx, ty, tz = -2.3, -0.08, 4.09 
    # roll, pitch, yaw = 0, -90, -90


    # Use this initialization for the VR app (first floor ARTLABS)
    tx, ty, tz = -25.3, -0.08, 5.09 
    roll, pitch, yaw = 0, -180, -90

    cam.set_camera_viewpoint(tx, ty, tz, roll, pitch, yaw)

    return cam

def splat_app_main(sh_degree, ply_file_path, out_dir):
    
    gaussian_model = GaussianModel(sh_degree, ply_file_path)

    cam = init_cam()
    
    #record_mode = Record_Mode.CONTINUE
    record_mode = Record_Mode.PAUSE

    recorder = Recorder(out_dir, record_mode)
    
    splat_app = SPLAT_APP(cam, gaussian_model, recorder)


    # First interactively record trajectory
    splat_app.vr_walkthrough_pygame(record_mode)
    print("Walk through is done... Replay with noise now....")

    #Define noise parameters for replaying training/recorded trajectories with noise injected
    # splat_app.trans_noise=0.025
    # splat_app.rot_noise_deg=1.25
    # splat_app.replay_with_noise(suffix="noisy1")
    # print("Replay with noise done.... Replaying another time with more noise...")

    # splat_app.trans_noise=0.05
    # splat_app.rot_noise_deg=2.5
    # splat_app.replay_with_noise(suffix="noisy2")
    # print("Replay with noise done.... Replaying another time with more noise...")


    # splat_app.trans_noise=0.075
    # splat_app.rot_noise_deg=3.75
    # splat_app.replay_with_noise(suffix="noisy3")
    # print("Replay with noise done.... Replaying another time with more noise...")

    # splat_app.trans_noise=0.1
    # splat_app.rot_noise_deg=5.0
    # splat_app.replay_with_noise(suffix="noisy4")
    # print("Replay with noise done.... Replaying another time with more noise...")


    # # Option 3: Replay training poses with noise
    # training_pose_file = "/root/code/extra1/datasets/ARTGarage/xgrids/1/ResultDataArtGarage_sample_2025-07-17-121502_0/ArtGarage_sample_2025-07-17-121502/img_traj.csv"
    # splat_app.replay_with_noise(training_pose_file)


def calculate_gaussian_feature_field_main(sh_degree, ply_file_path, out_dir):


    gaussian_model = GaussianModel(sh_degree, ply_file_path)

    cam = init_cam()
    
    #record_mode = Record_Mode.CONTINUE
    record_mode = Record_Mode.PAUSE

    recorder = Recorder(out_dir, record_mode)
    
    splat_app = SPLAT_APP(cam, gaussian_model, recorder)


    # First segment splat image directory and then transfer the semantics to the 3DGS model
    if 1:
        splat_segmenter = SplatSegmenter(out_dir)

        splat_segmenter.segment_splat_image_dir(record_mode=True)

        del splat_segmenter
    else:
        splat_app.transfer_saved_features_to_splat(out_dir)



# ------------------------------
# Training loop
# ------------------------------

def train_autoencoder(
    feature_dir,
    latent_dim=32,
    epochs=10,
    batch_size=1024,
    lr=1e-3,
    sample_size=100_000,
    use_mixed_precision=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CLIPFeatureDataset(feature_dir, sample_size=sample_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = CLIPAutoencoder(input_dim=512, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                rec, _ = model(batch)
                loss = criterion(rec, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"🧠 Epoch {epoch+1}/{epochs} | Avg MSE Loss: {avg_loss:.6f}")

        # Save checkpoint every epoch
        ckpt_path = os.path.join(feature_dir, f"autoencoder_{latent_dim}d_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(feature_dir, f"autoencoder_{latent_dim}d_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training complete! Saved final model to {final_path}")

    return model


# ------------------------------
# Helper: encode & save compressed features
# ------------------------------

def encode_feature_maps(model, feature_dir, latent_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    latent_dir = os.path.join(feature_dir, f"latent_{latent_dim}d")
    os.makedirs(latent_dir, exist_ok=True)

    with torch.no_grad():
        for f in tqdm(sorted(glob.glob(os.path.join(feature_dir, "*.pt"))), desc="Encoding feature maps"):
            feat = torch.load(f, map_location=device)  # [H, W, 512]
            H, W, _ = feat.shape
            z = model.encoder(feat.view(-1, 512))
            z = z.view(H, W, latent_dim)
            save_path = os.path.join(latent_dir, os.path.basename(f))
            torch.save(z.half().cpu(), save_path)
    print(f"✅ Encoded latent features saved to {latent_dir}")



def edit_gaussians_main(sh_degree, ply_file_path, out_dir):
    
    gaussian_model = GaussianModel(sh_degree, ply_file_path)
    
    #record_mode = Record_Mode.CONTINUE
    record_mode = Record_Mode.PAUSE

    recorder = Recorder(out_dir, record_mode)

    cam = init_cam()

    record_multiview_poses = True

    try:
        # Load multiview_camera_poses recorded
        json_path = Path(out_dir) / "recorded_camera_poses.json" 
            
        with open(json_path, "r") as f:
            data = json.load(f)
            multiview_cam_poses = data["poses"]

        if len(multiview_cam_poses) == 0:
            print("⚠️ recorded_camera_poses.json has no poses!")
            record_multiview_poses = True
        else:
            for pose in multiview_cam_poses:
                cam.recorded_poses.append(pose)
            record_multiview_poses = False

    except FileNotFoundError:
        record_multiview_poses = True

    splat_app = SPLAT_APP(cam, gaussian_model, recorder)

    # Walkthrough the splat to assess quality of edit
    if record_multiview_poses or True:
        splat_app.vr_walkthrough_pygame(record_mode)
        print("Walk through is done... Replay with noise now....")


    splat_app.select_object_gaussians_multiview()

    # splat_app.selected_gaussians = np.array(list(splat_app.selected_gaussians), dtype=np.int32)

    # if 1:
    #     splat_app.remove_selected_gaussians()
    # else:
    #     splat_app.shift_selected_gaussians(translation=(1.0, 0.0, 0.5))


    # splat_app.select_object_interactively_v2()

    
    # # Walkthrough the splat to assess quality of edit
    # splat_app.vr_walkthrough_pygame(record_mode)
    # print("Walk through is done... Replay with noise now....")


    # if 0:
    #     # Select Pixel Interactively
    #     splat_app.select_pixel_interactively()
    # else:
    #     # Select Object Interactively
    #     splat_app.select_object_interactively_v2()

    # if 1:
    #     # Remove Selected Gaussians
    #     splat_app.remove_selected_gaussians()
    # else:
    #     # Move Selected Gaussians
    #     splat_app.shift_selected_gaussians(translation=(1.0, 0.0, 0.5))


    # # Debug render to check editing quality
    # splat_app.debug_render_original_and_edited()



def main():

    # sh_degree = 3
    # ply_file_path="/root/code/extra1/datasets/ARTGarage/xgrids/4/Gaussian/PLY_Generic_splats_format/point_cloud/iteration_100/point_cloud.ply"

    sh_degree = 3
    #ply_file_path="/root/code/datasets/xgrids/LCC_output/AG_Office/ply-result/point_cloud/iteration_100/point_cloud.ply"
    ply_file_path="/root/code/ubuntu_data/datasets/ARTGarage/lab_office_in_out_k1_scanner/output/LCC_Studio_GaussianSplat_out/AG_Office/ply-result/point_cloud/iteration_100/point_cloud.ply"
    #ply_file_path="/root/code/ubuntu_data/datasets/ARTGarage/ARTLabs/output/ply-result/point_cloud/iteration_100/point_cloud.ply"


    # sh_degree = 3
    # ply_file_path="/root/code/datasets/ARTGarage/lab_office_in_out_k1_scanner/LCC_Studio_GaussianSplat_out/AG_lab/ply-result/point_cloud/iteration_100/point_cloud.ply"

    #sh_degree = 0
    #ply_file_path="/root/code/datasets/xgrids/LCC_output/portal_cam_output_LCC/output/ply-result/point_cloud/iteration_100/point_cloud_1.ply"

    out_dir = Path("/root/code/output/segmentation_testing/")

    # First perform VR walk through (disable segmentation inside vr walkthrough rasterizer as well) and record a trajectory
    # Then run calculate gaussian feature field for segmenting the images recorded
    # Disable feature field calculation when segmentation is being run if you want
    # Run AnyLabel to improve annotation performance
    # Enable feature field calculation if you want to lift the segmentation into the Gaussians via backpropagation

    # if 1:
    #     splat_app_main(sh_degree, ply_file_path, out_dir)
    # else:
    #     calculate_gaussian_feature_field_main(sh_degree, ply_file_path, out_dir)


    edit_gaussians_main(sh_degree, ply_file_path, out_dir)


if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser(description="Train Autoencoder on CLIP feature maps")
    # parser.add_argument("--feature_dir", type=str, required=True, help="Path to folder with .pt feature maps")
    # parser.add_argument("--latent_dim", type=int, default=32, help="Latent space dimensionality")
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--batch_size", type=int, default=1024)
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--sample_size", type=int, default=100_000)
    # parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")

    # args = parser.parse_args()

    # model = train_autoencoder(
    #     args.feature_dir,
    #     latent_dim=args.latent_dim,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     lr=args.lr,
    #     sample_size=args.sample_size,
    #     use_mixed_precision=not args.no_amp,
    # )

    # # Optional: encode all maps into latent representations
    # encode_feature_maps(model, args.feature_dir, args.latent_dim)
