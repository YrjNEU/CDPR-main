# import os
# import numpy as np
# import cv2
# import math
# import shutil
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--source_folder", type=str, required=False, default='/media/neu/YINGJIE/hammer/raw_data')
# parser.add_argument("--output_folder", type=str, required=False, default='/media/neu/YINGJIE/hammer/processed_data',
#                     help="Path to save processed data")
# args = parser.parse_args()
# dataroot = args.source_folder
# print(dataroot)
# output_folder = args.output_folder
# print(output_folder)
#
# if not os.path.exists(dataroot):
#     print("Source data folder ({}) does not exist! Please copy polarization data from original hammer first!".format(dataroot))
#     exit(1)
#
# # 检查输出文件夹是否存在，如果不存在则创建
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# print("Processing data from ({})".format(dataroot))
#
# import os
# import numpy as np
# import cv2
#
#
# def process_polar(pol_path):
#     polar_raw = cv2.imread(pol_path, -1)
#
#     H = polar_raw.shape[0]
#     W = polar_raw.shape[1]
#
#     # 通过猜测方式进行偏振角度映射
#     pol_0 = polar_raw[0:H // 2, 0:W // 2, :]  # top left - 0
#     pol_45 = polar_raw[0:H // 2, W // 2:, :]  # top right - 45
#     pol_90 = polar_raw[H // 2:, 0:W // 2:, :]  # bottom left - 90
#     pol_135 = polar_raw[H // 2:, W // 2:, :]  # bottom right - 135
#
#     # 将每个偏振角度下的图像转换为灰度图
#     pol_0, pol_45, pol_90, pol_135 = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, :, None].astype(np.float64) for im in
#                                       [pol_0, pol_45, pol_90, pol_135]]
#
#     pol_0 = pol_0 / 255
#     pol_45 = pol_45 / 255
#     pol_135 = pol_135 / 255
#     pol_90 = pol_90 / 255
#
#     # 计算偏振表示
#     I = (pol_0 + pol_45 + pol_90 + pol_135) / 2.
#     Q = (pol_0 - pol_90).astype(np.float64)
#     U = (pol_45 - pol_135).astype(np.float64)
#     Q[Q == 0] = 1e-10
#     I[I == 0] = 1e-10
#     rho = (np.sqrt(np.square(Q) + np.square(U)) / I).clip(0, 1)
#     phi = 0.5 * np.arctan(U / Q)
#     cos_2phi = np.cos(2 * phi)
#     check_sign = cos_2phi * Q
#     phi[check_sign < 0] = phi[check_sign < 0] + math.pi / 2.
#     phi = (phi + math.pi) % math.pi
#
#     pol_reps = np.concatenate([rho, phi], axis=2)
#
#     return pol_reps
#
#
# data_list = sorted(os.listdir(dataroot))
#
# for i in data_list:
#     if not ("scene" in i and "traj" in i and (not "naked" in i)):
#         continue
#
#     idx = int(i.split('scene')[1].split('_tra')[0])
#     data_path = os.path.join(dataroot, i)
#     data_pol_path = os.path.join(data_path, 'polarization')
#
#     sub_list = sorted(os.listdir(data_pol_path))
#     print('data_pol_path', data_pol_path)
#     print(sub_list)
#
#     for i in data_list:
#         if not ("scene" in i and "traj" in i and (not "naked" in i)):
#             continue
#
#         idx = int(i.split('scene')[1].split('_tra')[0])
#         data_path = os.path.join(dataroot, i)
#         data_pol_path = os.path.join(data_path, 'polarization', 'pol')  # 直接进入 pol 文件夹
#
#         if not os.path.isdir(data_pol_path):
#             continue
#
#         sub_list = sorted(os.listdir(data_pol_path))
#         print('data_pol_path', data_pol_path)
#         print('Found images:', sub_list)
#
#         for j in sub_list:
#             if not j.endswith('.png'):
#                 continue  # 跳过非png文件
#
#             pol_path = os.path.join(data_pol_path, j)
#             print('pol_path:', pol_path)
#
#             # 设置保存路径
#             new_pol_dir = os.path.join(output_folder, os.path.basename(data_path), 'pol_processed')
#             os.makedirs(new_pol_dir, exist_ok=True)
#
#             new_pol_path = os.path.join(new_pol_dir, j.replace('.png', '.npy'))
#             print('Saving to:', new_pol_path)
#
#             # 处理并保存
#             pol_reps = process_polar(pol_path)
#             np.save(new_pol_path, pol_reps)
#             print(f"Saved processed polarization data to {new_pol_path}")
#             # exit(0)

import os
import numpy as np
import cv2
import math
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=str, required=False, default='/media/neu/YINGJIE/hammer/raw_data')
parser.add_argument("--output_folder", type=str, required=False, default='/media/neu/YINGJIE/hammer/processed_data',
                    help="Path to save processed data")
args = parser.parse_args()
dataroot = args.source_folder
output_folder = args.output_folder

print(f"Source: {dataroot}")
print(f"Output: {output_folder}")

if not os.path.exists(dataroot):
    print(f"Source data folder ({dataroot}) does not exist!")
    exit(1)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def process_polar(pol_path):
    polar_raw = cv2.imread(pol_path, -1)
    H = polar_raw.shape[0]
    W = polar_raw.shape[1]

    pol_0 = polar_raw[0:H // 2, 0:W // 2, :]
    pol_45 = polar_raw[0:H // 2, W // 2:, :]
    pol_90 = polar_raw[H // 2:, 0:W // 2, :]
    pol_135 = polar_raw[H // 2:, W // 2:, :]

    pol_0 = cv2.cvtColor(pol_0, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    pol_45 = cv2.cvtColor(pol_45, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    pol_90 = cv2.cvtColor(pol_90, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    pol_135 = cv2.cvtColor(pol_135, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    I = (pol_0 + pol_45 + pol_90 + pol_135) / 2.
    Q = pol_0 - pol_90
    U = pol_45 - pol_135
    Q[Q == 0] = 1e-10
    I[I == 0] = 1e-10

    rho = (np.sqrt(np.square(Q) + np.square(U)) / I).clip(0, 1)
    phi = 0.5 * np.arctan(U / Q)
    cos_2phi = np.cos(2 * phi)
    check_sign = cos_2phi * Q
    phi[check_sign < 0] += math.pi / 2.
    phi = (phi + math.pi) % math.pi

    return rho, phi

data_list = sorted(os.listdir(dataroot))

for i in data_list:
    if not ("scene" in i and "traj" in i and "naked" not in i):
        continue

    data_path = os.path.join(dataroot, i)
    data_pol_path = os.path.join(data_path, 'polarization', 'pol')
    data_rgb_path = os.path.join(data_path, 'polarization', 'rgb')
    data_gt_path = os.path.join(data_path, 'polarization', '_gt')

    if not os.path.isdir(data_pol_path):
        continue

    sub_list = sorted(os.listdir(data_pol_path))
    print(f"[{i}] Found {len(sub_list)} pol images")

    for j in sub_list:
        if not j.endswith('.png'):
            continue

        base_name = os.path.splitext(j)[0]
        pol_path = os.path.join(data_pol_path, j)
        rgb_path = os.path.join(data_rgb_path, j)
        gt_path = os.path.join(data_gt_path, j)

        save_dir = os.path.join(output_folder, i, 'pol_processed')
        os.makedirs(save_dir, exist_ok=True)

        # 输出路径
        dolp_path = os.path.join(save_dir, base_name + '_dolp.png')
        aolp_path = os.path.join(save_dir, base_name + '_aolp.png')
        rgb_out_path = os.path.join(save_dir, base_name + '_rgb.png')
        gt_out_path = os.path.join(save_dir, base_name + '_depth.png')

        # 处理并保存 AoLP & DoLP
        rho, phi = process_polar(pol_path)
        dolp_8bit = (rho * 255).astype(np.uint8)
        aolp_8bit = ((phi / np.pi) * 255).astype(np.uint8)
        cv2.imwrite(dolp_path, dolp_8bit)
        cv2.imwrite(aolp_path, aolp_8bit)

        # 拷贝对应 RGB 和 GT 图像
        if os.path.exists(rgb_path):
            shutil.copy(rgb_path, rgb_out_path)
        else:
            print(f"[Warning] Missing RGB image: {rgb_path}")

        if os.path.exists(gt_path):
            shutil.copy(gt_path, gt_out_path)
        else:
            print(f"[Warning] Missing GT image: {gt_path}")

        print(f"Saved: {dolp_path}, {aolp_path}, {rgb_out_path}, {gt_out_path}")