# Author: Bingxin Ke
# Last modified: 2024-02-19
# 对原始hypersim数据集进行预处理：数据集划分、对RGB图像进行色调映射
# 对距离数据通过转换函数得到深度图，并处理无效像素，将处理后的RGB和深度图转换为png格式
# python script/dataset_preprocess/hypersim/preprocess_hypersim_pol.py
import argparse
import os
import cv2
import h5py
import numpy as np
import pandas as pd
from hypersim_util import dist_2_depth, tone_map
from tqdm import tqdm
# ====== 导入 component.py 中的 gen_mask ======
from pol_generator.component import gen_mask
# ====== 导入 utils.py 中的函数 ======
from pol_generator.utils import *
from pol_generator.normal2pol import generate_single_syn_daolp_by_labeled_type

IMG_WIDTH = 1024
IMG_HEIGHT = 768
FOCAL_LENGTH = 886.81

def readhdf5(hdf5_path):
    """ 简易封装从HDF5中读出dataset """
    with h5py.File(hdf5_path, "r") as f:
        data = np.array(f["dataset"])
    return data

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        type=str,
        default="/media/neu/YINGJIE/hypersim/metadata_images_split_scene_v1.csv",
    )
    parser.add_argument("--dataset_dir", type=str, default="/media/neu/YINGJIE/hypersim/data")
    parser.add_argument("--output_dir", type=str, default="/media/neu/YINGJIE/hypersim/processed")

    args = parser.parse_args()

    split_csv = args.split_csv
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # %%
    raw_meta_df = pd.read_csv(split_csv)
    meta_df = raw_meta_df[raw_meta_df.included_in_public_release].copy()

    # %%
    for split in ["train", "val", "test"]:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        split_meta_df = meta_df[meta_df.split_partition_name == split].copy()
        split_meta_df["rgb_path"] = None
        split_meta_df["rgb_mean"] = np.nan
        split_meta_df["rgb_std"] = np.nan
        split_meta_df["rgb_min"] = np.nan
        split_meta_df["rgb_max"] = np.nan
        split_meta_df["depth_path"] = None
        split_meta_df["depth_mean"] = np.nan
        split_meta_df["depth_std"] = np.nan
        split_meta_df["depth_min"] = np.nan
        split_meta_df["depth_max"] = np.nan
        split_meta_df["invalid_ratio"] = np.nan

        for i, row in tqdm(split_meta_df.iterrows(), total=len(split_meta_df)):
            # Load data
            scene_path = row.scene_name
            camera_name = row.camera_name
            frame_id = row.frame_id
            scene_id = os.path.join(split_output_dir, scene_path)
            os.makedirs(scene_id, exist_ok=True)
            rgb_path = os.path.join(
                dataset_dir,
                scene_path,
                "images",
                f"scene_{camera_name}_final_hdf5",
                f"frame.{frame_id:04d}.color.hdf5",
            )
            dist_path = os.path.join(
                dataset_dir,
                scene_path,
                "images",
                f"scene_{camera_name}_geometry_hdf5",
                f"frame.{frame_id:04d}.depth_meters.hdf5",
            )
            render_entity_id_path = os.path.join(
                dataset_dir,
                scene_path,
                "images",
                f"scene_{camera_name}_geometry_hdf5",
                f"frame.{frame_id:04d}.render_entity_id.hdf5",
            )
            normal_hdf5 = os.path.join(
                dataset_dir,
                scene_path,
                "images",
                f"scene_{camera_name}_geometry_hdf5",
                f"frame.{frame_id:04d}.normal_cam.hdf5",
            )
            # 用于gen_mask生成镜面掩码的三个HDF5:
            di_hdf5 = os.path.join(
                dataset_dir,
                scene_path,
                "images",
                f"scene_{camera_name}_final_hdf5",
                f"frame.{frame_id:04d}.diffuse_illumination.hdf5"
            )
            dr_hdf5 = os.path.join(
                dataset_dir,
                scene_path,
                "images",
                f"scene_{camera_name}_final_hdf5",
                f"frame.{frame_id:04d}.diffuse_reflectance.hdf5"
            )
            rs_hdf5 = os.path.join(
                dataset_dir,
                scene_path,
                "images",
                f"scene_{camera_name}_final_hdf5",
                f"frame.{frame_id:04d}.residual.hdf5"
            )

            assert os.path.exists(rgb_path)
            assert os.path.exists(dist_path)

            rgb = readhdf5(rgb_path).astype(float)
            dist_from_center = readhdf5(dist_path).astype(float)
            render_entity_id = readhdf5(render_entity_id_path).astype(int)
            normal = readhdf5(normal_hdf5).astype(float)
            diffuse_illum = readhdf5(di_hdf5).astype(float)
            diffuse_refl = readhdf5(dr_hdf5).astype(float)
            residual = readhdf5(rs_hdf5).astype(float)
            # 生成漫反射-镜面反射主导图，0：镜面反射，1：漫反射
            binary_img = gen_mask(
                diffuse_illum,
                diffuse_refl,
                residual,
                threshold=0.5,
                eps=1e-7,
                probability=False
            )

            # with h5py.File(os.path.join(dataset_dir, rgb_path), "r") as f:
            #     rgb = np.array(f["dataset"]).astype(float)
            # with h5py.File(os.path.join(dataset_dir, dist_path), "r") as f:
            #     dist_from_center = np.array(f["dataset"]).astype(float)
            # with h5py.File(os.path.join(dataset_dir, render_entity_id_path), "r") as f:
            #     render_entity_id = np.array(f["dataset"]).astype(int)

            # Tone map
            rgb_color_tm = tone_map(rgb, render_entity_id)
            rgb_int = (rgb_color_tm * 255).astype(np.uint8)  # [H, W, RGB]
            rgb_int = rgb_int[..., ::-1]


            # Distance -> depth
            plane_depth = dist_2_depth(
                IMG_WIDTH, IMG_HEIGHT, FOCAL_LENGTH, dist_from_center
            )
            valid_mask = render_entity_id != -1

            # 有效像素比例
            invalid_ratio = (np.prod(valid_mask.shape) - valid_mask.sum()) / np.prod(
                valid_mask.shape
            )
            plane_depth[~valid_mask] = 0

            # 法线预处理
            normal_img = normal.copy()
            normal_img = setNormal(normal_img)
            # 生成偏振图
            dp_st, ap_st = generate_single_syn_daolp_by_labeled_type(rgb_int, normal_img, binary_img)

            # # Save as png
            # if not os.path.exists(os.path.join(split_output_dir, scene_path)):
            #     os.makedirs(os.path.join(split_output_dir, scene_path))
            # 写入处理后的RGB图
            rgb_name = f"rgb_{camera_name}_fr{frame_id:04d}.png"
            rgb_path = os.path.join(scene_path, rgb_name)
            cv2.imwrite(
                os.path.join(split_output_dir, rgb_path),
                rgb_int
            )

            # 写入处理后的深度图
            plane_depth *= 1000.0
            plane_depth = plane_depth.astype(np.uint16)
            depth_name = f"depth_plane_{camera_name}_fr{frame_id:04d}.png"
            depth_path = os.path.join(scene_path, depth_name)
            cv2.imwrite(os.path.join(split_output_dir, depth_path), plane_depth)

            # 写入处理后的偏振图
            # AoLP
            aolp_name = f"aolp_{camera_name}_fr{frame_id:04d}.png"
            aolp_path = os.path.join(scene_path, aolp_name)
            cv2.imwrite(
                os.path.join(split_output_dir, aolp_path),
                ap_st
            )
            # DoLP
            dolp_name = f"dolp_{camera_name}_fr{frame_id:04d}.png"
            dolp_path = os.path.join(scene_path, dolp_name)
            cv2.imwrite(
                os.path.join(split_output_dir, dolp_path),
                dp_st
            )
            # Meta data
            split_meta_df.at[i, "rgb_path"] = rgb_path
            split_meta_df.at[i, "rgb_mean"] = np.mean(rgb_int)
            split_meta_df.at[i, "rgb_std"] = np.std(rgb_int)
            split_meta_df.at[i, "rgb_min"] = np.min(rgb_int)
            split_meta_df.at[i, "rgb_max"] = np.max(rgb_int)

            split_meta_df.at[i, "depth_path"] = depth_path
            restored_depth = plane_depth / 1000.0
            split_meta_df.at[i, "depth_mean"] = np.mean(restored_depth)
            split_meta_df.at[i, "depth_std"] = np.std(restored_depth)
            split_meta_df.at[i, "depth_min"] = np.min(restored_depth)
            split_meta_df.at[i, "depth_max"] = np.max(restored_depth)

            split_meta_df.at[i, "invalid_ratio"] = invalid_ratio

        with open(
            os.path.join(split_output_dir, f"filename_list_{split}.txt"), "w+"
        ) as f:
            lines = split_meta_df.apply(
                lambda r: f"{r['rgb_path']} {r['depth_path']}", axis=1
            ).tolist()
            f.writelines("\n".join(lines))

        with open(
            os.path.join(split_output_dir, f"filename_meta_{split}.csv"), "w+"
        ) as f:
            split_meta_df.to_csv(f, header=True)

    print("Preprocess finished")