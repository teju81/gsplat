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


        # print(f"x: {self.x}, y:{self.y}, z:{self.z}")

        return

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
            if frame_id > 20:
                break
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
        renders, alphas, info = self.gaussian_model.rasterize_splats(
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


        return colors, depths

    def rasterize_images_and_segment(self):

        # === Call the Gaussian Rasterizer ===
        colors, depths = self.rasterize_rgbd()

        # # Convert to CPU and numpy
        rendered_rgb_3dgs = colors[0].clamp(0, 1).detach().cpu().numpy()  # [H, W, 3]
        rendered_depth_3dgs = depths[0].squeeze(2).detach().cpu().numpy()  # [H, W]

        # === Convert to displayable format ===
        rgb_vis_3dgs = (rendered_rgb_3dgs * 255).astype(np.uint8)

        depth_min = rendered_depth_3dgs.min()
        depth_max = rendered_depth_3dgs.max()
        depth_vis_3dgs = normalize_depth(rendered_depth_3dgs, depth_min, depth_max)


        rgb_vis_3dgs_bgr = cv2.cvtColor(rgb_vis_3dgs, cv2.COLOR_RGB2BGR)
        
        image_pil = Image.fromarray(rgb_vis_3dgs_bgr) 
        image_transformed, _ = self.transform(image_pil, None)
        seg_img = self.splat_segmenter.langsam_gaussian_segmenter(rgb_vis_3dgs_bgr, image_transformed)
        seg_img = None

        return rgb_vis_3dgs_bgr, rgb_vis_3dgs, rendered_depth_3dgs, depth_vis_3dgs, seg_img


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


    def vr_walkthrough_pygame(self, record_mode):

        pygame.init()
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

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
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


                # elif event.type == pygame.MOUSEMOTION:
                #     handle_mouse_input(event)

            self.cam.pygame_move_camera()

            pose = self.cam.T.squeeze(0).detach().cpu().numpy().astype(np.float32)

            rgb_vis_3dgs_bgr, rgb_vis_3dgs, rendered_depth_3dgs, depth_vis_3dgs, seg_img = self.rasterize_images_and_segment()

            if record_mode in [Record_Mode.RECORD, Record_Mode.CONTINUE]:
                #self.recorder.record(rgb=rgb_vis_3dgs_bgr, depth=rendered_depth_3dgs, norm_depth=depth_vis_3dgs, pose=pose, seg=seg_img)
                self.recorder.record(rgb=rgb_vis_3dgs_bgr, depth=rendered_depth_3dgs, norm_depth=depth_vis_3dgs, pose=pose)

            if screen_capture:
                self.recorder.screen_capture(rgb=rgb_vis_3dgs_bgr, depth=rendered_depth_3dgs, norm_depth=depth_vis_3dgs, pose=pose, Ks=self.cam.Ks)
                screen_capture = False


            # === Display the output image ===
            # Convert the NumPy array to a Pygame surface
            # The swapaxes() is crucial because NumPy arrays and Pygame surfaces have different memory layouts.
            # NumPy is (height, width, channels), Pygame is (width, height, channels).

            rgb_game_img = rgb_vis_3dgs.copy()
            cv2.putText(rgb_game_img, f"Record Mode: {record_mode}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if show_help_menu:
                cv2.putText(rgb_game_img, "Help Menu - Keyboard Shortcuts", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(rgb_game_img, "C - For Screenshot of Current Camera Render", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "WASD - For Translational Camera Control", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "IK - YAW Camera Control", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "JL - Pitch Camera Control", (100, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(rgb_game_img, "UO - Roll Camera Control", (100, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


            image_surface = pygame.surfarray.make_surface(rgb_game_img.swapaxes(0, 1))

            # Blit the surface to the screen
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

            colors, depths = self.rasterize_rgbd()

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

    # def replay_from_training_poses(self, gaussian_model, pose_file, out_dir="train_replay",
    #                                file_type="csv", trans_noise=0.05, rot_noise_deg=2.0):



    #     poses = load_training_poses(pose_file, file_type)


    #     for frame_id, pose in poses.items():
    #         noisy_pose = perturb_pose(pose, trans_noise, rot_noise_deg)

    #         cam.T = torch.from_numpy(noisy_pose).unsqueeze(0).cuda()
    #         colors, depths = rasterize_rgbd(cam, gaussian_model)

    #         rendered_rgb = (colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    #         rendered_depth = depths[0].squeeze(2).cpu().numpy()
    #         norm_depth = normalize_depth(rendered_depth, rendered_depth.min(), rendered_depth.max())

    #         recorder.record(rendered_rgb, rendered_depth, norm_depth, pose, noisy_pose)

    #     recorder.save_json()
    #     print(f"✅ Training pose replay (with noise) saved to {out_dir}")


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


def init_cam():
    # Initialize Camera
    H = 1080
    W = 1920
    fx = fy = 1080
    near_plane, far_plane = 0.001, 100.0
    cam = Camera(H, W, fx, fy, near_plane, far_plane)
    

    # Track position and orientation of the camera over time
    tx, ty, tz, roll, pitch, yaw = 0, 0, 0, 0, 0, 0


    tx, ty, tz = 2, -1, 1
    roll, pitch, yaw = 0, 180, -90

    # Use this initialization for the VR app (first floor origin near stairs)
    # tx, ty, tz = -2.3, -0.08, 4.09 
    # roll, pitch, yaw = 0, -90, -90
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



def main():

    # sh_degree = 3
    # ply_file_path="/root/code/extra1/datasets/ARTGarage/xgrids/4/Gaussian/PLY_Generic_splats_format/point_cloud/iteration_100/point_cloud.ply"

    sh_degree = 3
    #ply_file_path="/root/code/datasets/xgrids/LCC_output/AG_Office/ply-result/point_cloud/iteration_100/point_cloud.ply"
    ply_file_path="/root/code/ubuntu_data/datasets/ARTGarage/lab_office_in_out_k1_scanner/output/LCC_Studio_GaussianSplat_out/AG_Office/ply-result/point_cloud/iteration_100/point_cloud.ply"
    ply_file_path="/root/code/ubuntu_data/datasets/ARTGarage/ARTLabs/output/ply-result/point_cloud/iteration_100/point_cloud.ply"


    # sh_degree = 3
    # ply_file_path="/root/code/datasets/ARTGarage/lab_office_in_out_k1_scanner/LCC_Studio_GaussianSplat_out/AG_lab/ply-result/point_cloud/iteration_100/point_cloud.ply"

    #sh_degree = 0
    #ply_file_path="/root/code/datasets/xgrids/LCC_output/portal_cam_output_LCC/output/ply-result/point_cloud/iteration_100/point_cloud_1.ply"

    out_dir = Path("/root/code/output/second_floor_trajectory/")


    if 0:
        splat_app_main(sh_degree, ply_file_path, out_dir)
    else:
        calculate_gaussian_feature_field_main(sh_degree, ply_file_path, out_dir)


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
