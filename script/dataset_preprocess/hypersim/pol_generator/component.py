import os.path

import h5py
import numpy as np
from PIL import Image  # 需要 pip install pillow



def gen_mask(
    diffuse_illum,
    diffuse_refl,
    residual,
    threshold,
    eps,
    probability
):
    """
    读取 3 个 HDF5 文件:
      - diffuse_illumination_path: 漫反射光照
      - diffuse_reflectance_path : 漫反射反射率
      - residual_path            : 非漫反射残差(如镜面高光等)
    计算:
      D = diffuse_illumination * diffuse_reflectance
      S = residual
      alpha = S / (D + S + eps)
    当 alpha > threshold => 镜面主导(1)，否则漫反射主导(0)。
    最后输出两份结果:
      1) HDF5: 保存在 hdf5_output_path, Dataset 名为 'label'
      2) JPG : 保存在 jpg_output_path, 供快速可视化(黑白图: 0=>黑, 1=>白)

    参数:
      diffuse_illumination_path: str, 漫反射光照HDF5
      diffuse_reflectance_path : str, 漫反射反射率HDF5
      residual_path            : str, 非漫反射残差HDF5
      hdf5_output_path         : str, 生成的掩码HDF5输出路径
      jpg_output_path          : str, 生成的预览JPEG输出路径
      threshold                : float, 二值化阈值
      eps                      : float, 防止分母为0
    """

    # # 1) 读取 HDF5 文件
    # with h5py.File(diffuse_illumination_path, 'r') as f_illum:
    #     print("Top-level keys:", list(f_illum.keys()))
    #     diffuse_illum = f_illum['dataset'][:]  # 假设数据存储在 'data' 键
    # with h5py.File(diffuse_reflectance_path, 'r') as f_refl:
    #     diffuse_refl = f_refl['dataset'][:]
    # with h5py.File(residual_path, 'r') as f_res:
    #     residual = f_res['dataset'][:]


    # 2) 计算漫反射强度 D = diffuse_illumination × diffuse_reflectance
    D = diffuse_illum * diffuse_refl

    # 3) 将 (H, W, 3) 的 RGB 转灰度, 若本身就是 (H, W) 则跳过
    if D.ndim == 3 and D.shape[-1] == 3:
        D_gray = D.mean(axis=-1)
    else:
        D_gray = D

    if residual.ndim == 3 and residual.shape[-1] == 3:
        S_gray = residual.mean(axis=-1)
    else:
        S_gray = residual
    # 4) 计算镜面比例 alpha
    alpha = D_gray / (D_gray + S_gray + eps)
    print("Check for invalid values in alpha:", np.isnan(alpha).any(), np.isinf(alpha).any())
    alpha = np.clip(alpha, 0, 1)
    # 5) 生成二值掩码: > threshold => 1(镜面), 否则0(漫反射)

    if probability:
        random_vals = np.random.rand(*alpha.shape)
        label_mask = (random_vals < alpha).astype(np.uint8)
    else:
        label_mask = np.zeros_like(alpha, dtype=np.uint8)
        label_mask[alpha > threshold] = 1
    return label_mask
    # # =============== 输出 1: HDF5 文件 ===============
    # with h5py.File(hdf5_output_path, 'w') as f_out:
    #     f_out.create_dataset('dataset', data=label_mask)
    # print(f'[Info] HDF5 mask saved to {hdf5_output_path}')
    #
    # # =============== 输出 2: JPG 文件 ===============
    # # 将0/1的掩码映射到0/255形成黑白图 (0=>黑, 255=>白)
    # label_mask_255 = label_mask * 255
    # # 转成PIL图像, 并以灰度格式(L)保存
    # pil_img = Image.fromarray(label_mask_255, mode='L')
    # pil_img.save(jpg_output_path, 'JPEG')
    # print(f'[Info] Preview JPG saved to {jpg_output_path}')


if __name__ == "__main__":
    # 示例: 假设在当前目录下有这几个文件
    input_root = r'E:\hypersim\data\ai_001_001\images\scene_cam_00_final_hdf5'
    output_root = r'E:\hypersim\data\ai_001_001\images\output_temp'
    os.makedirs(output_root, exist_ok=True)
    # input_root = r'D:\python_work\ai_021_007\images\scene_cam_00_final_hdf5'
    # output_root = r'D:\python_work\ai_021_007\images\output_temp'
    di_path = os.path.join(input_root, 'frame.0000.diffuse_illumination.hdf5')
    dr_path = os.path.join(input_root, 'frame.0000.diffuse_reflectance.hdf5')
    rs_path = os.path.join(input_root, 'frame.0000.residual.hdf5')
    out_hdf5 = os.path.join(output_root, 'frame.0000.mirror_mask.hdf5')
    out_jpg = os.path.join(output_root, 'frame.0000.mirror_mask.jpg')
    # di_path = r'E:\hypersim\data\ai_021_007\images\scene_cam_00_final_hdf5\frame.0000.diffuse_illumination.hdf5'
    # dr_path = r'E:\hypersim\data\ai_021_007\images\scene_cam_00_final_hdf5\frame.0000.diffuse_reflectance.hdf5'
    # rs_path = r'E:\hypersim\data\ai_021_007\images\scene_cam_00_final_hdf5\frame.0000.residual.hdf5'
    # out_hdf5 = r'E:\hypersim\data\ai_021_007\images\output_temp\frame.0000.mirror_mask.hdf5'
    # out_jpg = r'E:\hypersim\data\ai_021_007\images\output_temp\frame.0000.mirror_mask.jpg'

    # 可以根据需要自定义threshold
    # 0:漫反射主导
    # 1:镜面反射主导
    gen_mask(
        diffuse_illumination_path=di_path,
        diffuse_reflectance_path=dr_path,
        residual_path=rs_path,
        hdf5_output_path=out_hdf5,
        jpg_output_path=out_jpg,
        threshold=0.5,
        eps=1e-7,
        probability=False
    )
    print('Done!')