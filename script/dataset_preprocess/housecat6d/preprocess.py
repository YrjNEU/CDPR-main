import argparse
import math
import os
import shutil

import cv2
import numpy as np

def process_polar(pol_path):
    polar_raw = cv2.imread(pol_path, -1)
    H, W = polar_raw.shape[:2]

    pol_0   = polar_raw[0:H//2, 0:W//2]
    pol_45  = polar_raw[0:H//2, W//2:]
    pol_90  = polar_raw[H//2:, 0:W//2]
    pol_135 = polar_raw[H//2:, W//2:]

    def to_gray(x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    I0   = to_gray(pol_0)
    I45  = to_gray(pol_45)
    I90  = to_gray(pol_90)
    I135 = to_gray(pol_135)

    I = (I0 + I45 + I90 + I135) / 2.0
    Q = I0 - I90
    U = I45 - I135

    Q[Q == 0] = 1e-10
    I[I == 0] = 1e-10

    rho = np.sqrt(Q * Q + U * U) / I
    rho = np.clip(rho, 0.0, 1.0)

    phi = 0.5 * np.arctan(U / Q)
    cos_2phi = np.cos(2 * phi)
    sign_check = cos_2phi * Q
    phi[sign_check < 0] += math.pi / 2.0
    phi = (phi + math.pi) % math.pi

    return rho, phi

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=str,
        default="/media/neu/YINGJIE/test_scene",
        help="HouseCat6D raw root containing test_scene*/",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/media/neu/YINGJIE/test_scene/processed",
        help="Directory to store processed HouseCat6D data.",
    )
    args = parser.parse_args()

    src_root = args.source_root
    out_root = args.output_root
    os.makedirs(out_root, exist_ok=True)

    txt_sparse = os.path.join(out_root, "test_sparse_depth.txt")
    txt_clean = os.path.join(out_root, "test_clean_depth.txt")
    sparse_lines = []
    clean_lines = []

    for scene in sorted(os.listdir(src_root)):
        if not scene.startswith("test_scene"):
            continue

        scene_dir = os.path.join(src_root, scene)
        if not os.path.isdir(scene_dir):
            continue

        pol_dir = os.path.join(scene_dir, "pol")
        rgb_dir = os.path.join(scene_dir, "rgb")
        depth_dir = os.path.join(scene_dir, "depth")
        clean_dir = os.path.join(scene_dir, "depth_gt")
        if not os.path.isdir(pol_dir):
            continue

        out_scene_dir = os.path.join(out_root, scene, "processed")
        os.makedirs(out_scene_dir, exist_ok=True)
        pol_files = sorted(f for f in os.listdir(pol_dir) if f.endswith(".png"))
        print(f"[{scene}] {len(pol_files)} frames")

        for fname in pol_files:
            base = os.path.splitext(fname)[0]
            pol_path = os.path.join(pol_dir, fname)
            rgb_path = os.path.join(rgb_dir, fname)
            sparse_dep = os.path.join(depth_dir, fname)
            clean_dep = os.path.join(clean_dir, fname)

            if not os.path.exists(rgb_path):
                print(f"Missing RGB: {rgb_path}")
                continue

            out_rgb = os.path.join(out_scene_dir, base + "_rgb.png")
            out_aolp = os.path.join(out_scene_dir, base + "_aolp.png")
            out_dolp = os.path.join(out_scene_dir, base + "_dolp.png")
            out_sparse = os.path.join(out_scene_dir, base + "_depth_sparse.png")
            out_clean = os.path.join(out_scene_dir, base + "_depth_clean.png")

            rho, phi = process_polar(pol_path)
            cv2.imwrite(out_dolp, (rho * 255).astype(np.uint8))
            cv2.imwrite(out_aolp, ((phi / math.pi) * 255).astype(np.uint8))
            shutil.copy(rgb_path, out_rgb)

            if os.path.exists(sparse_dep):
                shutil.copy(sparse_dep, out_sparse)
            else:
                out_sparse = None

            if os.path.exists(clean_dep):
                shutil.copy(clean_dep, out_clean)
            else:
                out_clean = None

            rel = lambda p: os.path.relpath(p, out_root)
            if out_sparse is not None:
                sparse_lines.append(
                    f"{rel(out_rgb)} {rel(out_sparse)} {rel(out_aolp)} {rel(out_dolp)}"
                )
            if out_clean is not None:
                clean_lines.append(
                    f"{rel(out_rgb)} {rel(out_clean)} {rel(out_aolp)} {rel(out_dolp)}"
                )

    with open(txt_sparse, "w") as f:
        f.write("\n".join(sparse_lines))
    with open(txt_clean, "w") as f:
        f.write("\n".join(clean_lines))

    print("Done.")
    print(f"Sparse depth txt: {txt_sparse}")
    print(f"Clean depth txt : {txt_clean}")
