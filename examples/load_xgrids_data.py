# load_xgrids_data.py
import numpy as np
from pathlib import Path

def parse_cameras_txt(cameras_txt):
    cameras = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            elems = line.strip().split()
            cam_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = list(map(float, elems[4:]))
            if model != "PINHOLE":
                raise NotImplementedError(f"Unsupported camera model: {model}")
            fx, fy, cx, cy = params
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)
            cameras[cam_id] = {"K": K, "width": width, "height": height}
    return cameras

def qvec2rotmat(qvec):
    q0, q1, q2, q3 = qvec
    R = np.array([
        [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1*q1 + q2*q2)]
    ], dtype=np.float32)
    return R

def parse_images_txt(images_txt):
    """Parse COLMAP-style images.txt (two lines per image)."""
    images = []
    with open(images_txt, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or len(line) == 0:
            i += 1
            continue

        elems = line.split()
        if len(elems) < 10:
            i += 1
            continue

        # First line of the image block
        img_id = int(elems[0])
        qvec = np.array(list(map(float, elems[1:5])))
        tvec = np.array(list(map(float, elems[5:8])))
        cam_id = int(elems[8])
        name = elems[9]

        R = qvec2rotmat(qvec)
        T = tvec

        images.append({
            "id": img_id,
            "cam_id": cam_id,
            "R": R,
            "T": T,
            "name": name
        })

        # Skip the following POINTS2D line
        i += 2

    return images


def parse_points3D_txt(points3D_txt):
    points, colors = [], []
    with open(points3D_txt, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            elems = line.strip().split()
            X, Y, Z = map(float, elems[1:4])
            R, G, B = map(int, elems[4:7])
            points.append([X, Y, Z])
            colors.append([R/255.0, G/255.0, B/255.0])
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)

def load_xgrids_scene(scene_dir):
    scene_dir = Path(scene_dir)
    cameras = parse_cameras_txt(scene_dir / "cameras.txt")
    images = parse_images_txt(scene_dir / "images.txt")
    pts3d, colors = parse_points3D_txt(scene_dir / "points3D.txt")
    return cameras, images, pts3d, colors
