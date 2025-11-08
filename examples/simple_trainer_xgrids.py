# simple_trainer_xgrids.py
"""
Train a 3DGS directly from Xgrids (COLMAP-style) export:
  cameras.txt, images.txt, points3D.txt, images/

Usage:
  python3 simple_trainer_xgrids.py --scene_dir /path/to/xgrids
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
import imageio.v3 as iio
import numpy as np
import struct
import threading

# --- gsplat rasterizer import ---
try:
    from gsplat.rendering import rasterization as gs_rasterize
except Exception:
    from gsplat import rasterization as gs_rasterize

# ✅ Try importing the visualizer (safe fallback if not available)
try:
    from gsplat.visualizer import Visualizer
    HAS_VIEWER = True
except Exception:
    print("⚠️  gsplat.visualizer not found. Proceeding without visualization.")
    HAS_VIEWER = False

from load_xgrids_data import load_xgrids_scene


# ----------------------------------------------------------------------
# 🔧 Replacement for utils.graphics_utils
# ----------------------------------------------------------------------
def getWorld2View2(R, T):
    """Build a 4x4 world-to-camera view matrix from rotation & translation."""
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.T  # inverse rotation
    Rt[:3, 3] = -R.T @ T
    return torch.tensor(Rt, dtype=torch.float32)


def getProjectionMatrix2(K, znear=0.01, zfar=50.0, width=None, height=None):
    """Convert intrinsic K to a normalized device coordinate projection matrix."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if width is None:
        width = int(2 * cx)
    if height is None:
        height = int(2 * cy)

    proj = torch.zeros((4, 4), dtype=torch.float32)
    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = 1 - (2 * cx / width)
    proj[1, 2] = (2 * cy / height) - 1
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    proj[3, 2] = -1
    return proj


# ----------------------------------------------------------------------
# 💾 Save PLY utility
# ----------------------------------------------------------------------
def save_3dgs_ply(path, means, scales, rotations, colors, opacities,
                  min_bounds=None, max_bounds=None):
    """Save 3DGS as binary little endian PLY with INRIA-compatible header."""
    N = means.shape[0]

    if min_bounds is None:
        min_bounds = means.detach().min(0).values.cpu().numpy()
    if max_bounds is None:
        max_bounds = means.detach().max(0).values.cpu().numpy()

    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
comment minx {min_bounds[0]:.7f}
comment miny {min_bounds[1]:.7f}
comment minz {min_bounds[2]:.7f}
comment maxx {max_bounds[0]:.7f}
comment maxy {max_bounds[1]:.7f}
comment maxz {max_bounds[2]:.7f}
comment offsetx 0.0000000000000000
comment offsety 0.0000000000000000
comment offsetz 0.0000000000000000
comment shiftx 0.0000000000000000
comment shifty 0.0000000000000000
comment shiftz 0.0000000000000000
comment scalex 1.0000000000000000
comment scaley 1.0000000000000000
comment scalez 1.0000000000000000
comment source K1
comment epsg 0
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
""" + "\n".join([f"property float f_rest_{i}" for i in range(45)]) + """
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))
        means = means.detach().cpu().numpy().astype(np.float32)
        scales = scales.detach().cpu().numpy().astype(np.float32)
        rotations = rotations.detach().cpu().numpy().astype(np.float32)
        colors = colors.detach().cpu().numpy().astype(np.float32)
        opacities = opacities.detach().cpu().numpy().astype(np.float32)

        zeros_rest = np.zeros(45, dtype=np.float32)
        nx, ny, nz = np.float32(0), np.float32(0), np.float32(1)

        for i in range(N):
            data = (
                means[i, 0], means[i, 1], means[i, 2],
                nx, ny, nz,
                colors[i, 0], colors[i, 1], colors[i, 2],
                *zeros_rest,
                opacities[i, 0],
                scales[i, 0], scales[i, 1], scales[i, 2],
                rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3],
            )
            f.write(struct.pack("<" + "f" * len(data), *data))
    print(f"✅ Saved PLY: {path}")


# ----------------------------------------------------------------------
# 🧠 Rasterizer wrapper
# ----------------------------------------------------------------------
def _call_rasterizer(means, scales, quats, opacities, colors,
                     R, T, K, H, W, device,
                     near_plane=0.01, far_plane=50.0):
    """Wrapper for gsplat.rasterization() using the modern API."""
    R = R.to(dtype=torch.float32)
    T = T.to(dtype=torch.float32)
    K = K.to(dtype=torch.float32)

    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = R.T
    viewmat[:3, 3] = -R.T @ T

    viewmats = viewmat.unsqueeze(0).unsqueeze(0)
    Ks = K.unsqueeze(0).unsqueeze(0)
    quats = quats.unsqueeze(0)
    means = means.unsqueeze(0)
    scales = scales.unsqueeze(0)
    colors = colors.unsqueeze(0)
    opacities = opacities.squeeze(-1).unsqueeze(0)

    rgb, depth, extras = gs_rasterize(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        near_plane=near_plane,
        far_plane=far_plane,
        backgrounds=torch.zeros(3, device=device),
        render_mode="RGB",
        camera_model="pinhole",
    )
    return rgb[0]


# ----------------------------------------------------------------------
# 🏋️‍♂️ Training loop
# ----------------------------------------------------------------------
def train_xgrids(scene_dir: str, lr: float = 1e-3, epochs: int = 50, znear: float = 0.01, zfar: float = 50.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🟢 Using device: {device}")

    cameras, images, pts3d, colors_np = load_xgrids_scene(scene_dir)
    print(f"Loaded {len(images)} images, {len(pts3d)} 3D points")

    means = torch.tensor(pts3d, device=device, dtype=torch.float32, requires_grad=True)
    colors = torch.tensor(colors_np, device=device, dtype=torch.float32, requires_grad=True)
    N = means.shape[0]
    scales = torch.full((N, 3), 0.01, device=device, dtype=torch.float32, requires_grad=True)
    rotations = torch.zeros((N, 4), device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    rotations.requires_grad_(True)
    opacities = torch.full((N, 1), 0.1, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = Adam([means, colors, scales, rotations, opacities], lr=lr)

    # ✅ Initialize the viewer (non-blocking)
    viewer = None
    if HAS_VIEWER:
        viewer = Visualizer(window_name="Xgrids 3DGS Trainer", width=960, height=720)
        viewer_thread = threading.Thread(target=viewer.show, daemon=True)
        viewer_thread.start()
        viewer.update(means.detach().cpu(), scales.detach().cpu(),
                      rotations.detach().cpu(), colors.detach().cpu(),
                      opacities.detach().cpu())

    for epoch in range(epochs):
        total_loss, valid = 0.0, 0
        for frame in images:
            cam = cameras[frame["cam_id"]]
            K = torch.tensor(cam["K"], device=device)
            R = torch.tensor(frame["R"], device=device)
            T = torch.tensor(frame["T"], device=device)
            H, W = cam["height"], cam["width"]

            img_path = os.path.join(scene_dir, "images", frame["name"])
            if not os.path.exists(img_path):
                print(f"⚠️  Missing image: {img_path}")
                continue

            gt = torch.tensor(iio.imread(img_path), device=device, dtype=torch.float32) / 255.0
            gt = gt[..., :3]
            if gt.shape[0] != H or gt.shape[1] != W:
                gt = F.interpolate(
                    gt.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0).permute(1, 2, 0)

            rendered = _call_rasterizer(means, scales, rotations, opacities,
                                        colors, R, T, K, H, W, device, znear, zfar)

            loss = F.mse_loss(rendered, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            valid += 1

        if valid:
            print(f"[Epoch {epoch:03d}] Avg Loss: {total_loss/valid:.6f}")
        else:
            print(f"[Epoch {epoch:03d}] No frames")

        # ✅ Update visualization every epoch
        if viewer and epoch % 1 == 0:
            viewer.update(means.detach().cpu(), scales.detach().cpu(),
                          rotations.detach().cpu(), colors.detach().cpu(),
                          opacities.detach().cpu())

    # Save trained model
    out_path = os.path.join(scene_dir, "xgrids_refined_3dgs.pt")
    torch.save(
        {"means": means.detach().cpu(),
         "colors": colors.detach().cpu(),
         "scales": scales.detach().cpu(),
         "rotations": rotations.detach().cpu(),
         "opacities": opacities.detach().cpu()},
        out_path,
    )
    print(f"✅ Saved model: {out_path}")

    # Export PLY
    ply_path = os.path.join(scene_dir, "xgrids_refined_3dgs.ply")
    save_3dgs_ply(ply_path, means, scales, rotations, colors, opacities)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", default="/root/code/ubuntu_data/datasets/ARTGarage/xgrids_sample_dataset/1/ResultDataArtGarage_sample_2025-07-17-121502_0/ArtGarage_sample_2025-07-17-121502/perspective")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--znear", type=float, default=0.01)
    ap.add_argument("--zfar", type=float, default=50.0)
    args = ap.parse_args()


    train_xgrids(args.scene_dir, args.lr, args.epochs, args.znear, args.zfar)
