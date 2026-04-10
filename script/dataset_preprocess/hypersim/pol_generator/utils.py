import numpy as np
import cv2
import math
import os
import pdb

eta = 1.5
input_root = r'D:\python_work\ai_021_007\images'
output_root = r'D:\python_work\ai_021_007\images\output_temp'

def dolp_aolp_to_polars(dolp_img, aolp_img, intensity_img):
    """
    根据 DoLP (dolp_img) 与 AoLP (aolp_img) 以及 intensity_img 生成 [I0, I45, I90, I135]。
    若 intensity_img 单通道(H, W)，返回4个单通道结果 (H, W, float32)。
    若 intensity_img 三通道(H, W, 3)，则对每个通道并行运算，返回4个 (H, W, 3, float32)。

    公式:
      I0   = Imean * [ 1 + rho * cos(2phi) ]
      I45  = Imean * [ 1 + rho * sin(2phi) ]
      I90  = Imean * [ 1 - rho * cos(2phi) ]
      I135 = Imean * [ 1 - rho * sin(2phi) ]

    返回:
      若单通道输入:  [I0, I45, I90, I135]，每个shape=(H, W)
      若三通道输入:  [I0, I45, I90, I135]，每个shape=(H, W, 3)
      均为 float32
    """

    # 为了向量化，需要保证 dolp_img, aolp_img 与 intensity_img 形状可广播
    # dolp_img, aolp_img: (H, W) float
    # intensity_img: (H, W) or (H, W, 3)

    # 单通道情况: (H, W)
    if intensity_img.ndim == 2:
        # 广播形状: (H, W)
        Imean = intensity_img.astype(np.float32)
        rho = dolp_img.astype(np.float32)
        phi = aolp_img.astype(np.float32)

        I0 = Imean * (1 + rho * np.cos(2 * phi))
        I45 = Imean * (1 + rho * np.sin(2 * phi))
        I90 = Imean * (1 - rho * np.cos(2 * phi))
        I135 = Imean * (1 - rho * np.sin(2 * phi))

        return [I0.astype(np.float32),
                I45.astype(np.float32),
                I90.astype(np.float32),
                I135.astype(np.float32)]

    # 三通道情况: (H, W, 3)
    elif intensity_img.ndim == 3 and intensity_img.shape[2] == 3:
        # 将 dolp_img, aolp_img 扩张维度 => (H, W, 1)，再与 (H, W, 3) 广播
        Imean = intensity_img.astype(np.float32)               # (H, W, 3)
        rho   = dolp_img.astype(np.float32)[..., None]         # (H, W, 1)
        phi   = aolp_img.astype(np.float32)[..., None]         # (H, W, 1)

        # 公式向量化
        I0   = Imean * (1 + rho * np.cos(2 * phi))
        I45  = Imean * (1 + rho * np.sin(2 * phi))
        I90  = Imean * (1 - rho * np.cos(2 * phi))
        I135 = Imean * (1 - rho * np.sin(2 * phi))

        return [
            I0.astype(np.float32),
            I45.astype(np.float32),
            I90.astype(np.float32),
            I135.astype(np.float32)
        ]

    else:
        raise ValueError("intensity_img must be either (H, W) or (H, W, 3).")


def setNormal(norm_gt_img):
    # 假设输入是浮点数类型，范围在 [0,1]
    norm_gt_img = norm_gt_img.astype(np.float32)
    # 映射到 [-1,1]
    norm_img = norm_gt_img * 2 - 1
    # 交换 x 与 z 通道
    normal_img = norm_img.copy()
    normal_img[..., 0] = norm_img[..., 2]
    normal_img[..., 2] = norm_img[..., 0]
    return normal_img

def PolarsToDolpAolp(polars_img):
    """
    将四幅极化图 (I0, I45, I90, I135) 转为 DoLP (dolp_img) 和 AoLP (aolp_img)。

    参数:
      polars_img: list/tuple 长度为4，每个元素是一个OpenCV图像(numpy数组)。
                  若通道数=3，先转换为灰度；若通道数=1，直接使用。
                  每幅图假定是8位(uint8)。

    返回:
      dolp_img: float32类型的灰度图
      aolp_img: float32类型的灰度图
    """
    # 1) 若4张图像是彩色，则转换为灰度
    polars_gray = []
    for i in range(4):
        img = polars_img[i]
        if img.ndim == 3 and img.shape[2] == 3:
            # BGR -> Gray
            gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
            polars_gray.append(gray)
        else:
            # Already single channel
            polars_gray.append(img.astype(np.float32))

    # 2) 分别取 I_0, I_45, I_90, I_135 (float32)
    I_0   = polars_gray[0]
    I_45  = polars_gray[1]
    I_90  = polars_gray[2]
    I_135 = polars_gray[3]

    # 可选：调试输出 i45 - i135 的最值、均值
    diff_45_135 = I_45 - I_135
    print("[DEBUG] i45 - i135 => min=%.3f, max=%.3f, mean=%.3f" %
          (diff_45_135.min(), diff_45_135.max(), diff_45_135.mean()))

    # 3) 计算 s0, s1, s2
    #    s0 = i0 + i90
    #    s1 = i0 - i90
    #    s2 = i45 - i135
    s0 = I_0 + I_90
    s1 = I_0 - I_90
    s2 = I_45 - I_135

    rows, cols = I_0.shape
    dolp_img = np.zeros((rows, cols), dtype=np.float32)
    aolp_img = np.zeros((rows, cols), dtype=np.float32)

    # 4) 构建mask处理特殊情况:
    #   - all_zero_mask: [i0,i45,i90,i135都为0]
    #   - near_zero_mask: [abs(s1)<1e-6,abs(s2)<1e-6]
    #   - s0_zero_mask: [s0=0]
    # 其余情况下计算正常公式

    # 全为0 => 无偏振信息
    all_zero_mask = (I_0 == 0) & (I_45 == 0) & (I_90 == 0) & (I_135 == 0)

    # 近似无偏振 => s1, s2近似0
    near_zero_mask = (np.abs(s1) < 1e-6) & (np.abs(s2) < 1e-6)

    # s0=0 => 不能除以0
    s0_zero_mask = (s0 == 0)

    # ========== 正常像素 ========== #
    valid_mask = ~(all_zero_mask | near_zero_mask | s0_zero_mask)

    # 5) 在 valid_mask 下计算 rho, phi
    #    rho = sqrt(s1^2 + s2^2) / s0
    #    phi = 0.5 * atan2(s2, s1)
    # 注意phi<0 => +pi
    rho = np.zeros_like(s0, dtype=np.float32)
    phi = np.zeros_like(s0, dtype=np.float32)

    rho[valid_mask] = np.sqrt(s1[valid_mask]**2 + s2[valid_mask]**2) / s0[valid_mask]
    phi[valid_mask] = 0.5 * np.arctan2(s2[valid_mask], s1[valid_mask])

    # phi < 0 => + pi
    negative_phi_mask = (phi < 0)
    phi[negative_phi_mask] += math.pi

    # 6) 写回 dolp_img, aolp_img
    dolp_img = rho
    aolp_img = phi

    return dolp_img, aolp_img


def normal2DAoLP(normal_img, eta=1.5):
    """
    将法向图 normal_img (H, W, 3) 向量化地转换为:
      - DDoLP_img, DAoLP_img: 漫反射(difffuse)的偏振度 & 偏振角
      - SDoLP_img, SAoLP_img: 镜面反射(specular)的偏振度 & 偏振角
      - zenith_img: 天顶角(0~pi)

    注意:
      1. 仅对中间区域 [1:-1, 1:-1] 做运算, 与原逻辑保持一致
      2. 如果 nz<0 => 翻转(nx, ny, nz)
      3. nz 限制在 [-1,1], 避免 acos 出错
      4. 最后将结果写回对应数组
    """
    H, W, _ = normal_img.shape
    DDoLP_img = np.zeros((H, W), dtype=np.float32)
    DAoLP_img = np.zeros((H, W), dtype=np.float32)
    SDoLP_img = np.zeros((H, W), dtype=np.float32)
    SAoLP_img = np.zeros((H, W), dtype=np.float32)
    zenith_img = np.zeros((H, W), dtype=np.float32)

    # 1) 提取中间区域 [1:-1,1:-1]，拷贝一份以进行翻转 / 算法处理
    sub_nx = normal_img[1:-1, 1:-1, 0].copy()
    sub_ny = normal_img[1:-1, 1:-1, 1].copy()
    sub_nz = normal_img[1:-1, 1:-1, 2].copy()

    # 2) 若 nz < 0, 则翻转 (nx, ny, nz)
    neg_mask = (sub_nz < 0)
    sub_nx[neg_mask] = -sub_nx[neg_mask]
    sub_ny[neg_mask] = -sub_ny[neg_mask]
    sub_nz[neg_mask] = -sub_nz[neg_mask]

    # 3) 将 nz 限幅 [-1,1], 计算 zenith, azimuth
    sub_nz = np.clip(sub_nz, -1.0, 1.0)
    sub_zenith = np.arccos(sub_nz)  # [0, pi]
    sub_azimuth = np.arctan2(sub_ny, sub_nx)  # 理论上 [-pi, pi]

    # 强行约束 azimuth ∈ (0, pi)
    mask_le0 = (sub_azimuth <= 0)
    sub_azimuth[mask_le0] += math.pi
    mask_gepi = (sub_azimuth >= math.pi)
    sub_azimuth[mask_gepi] -= math.pi

    # 将计算结果写回 zenith_img
    zenith_img[1:-1, 1:-1] = sub_zenith

    # 4) 计算 sin, cos 及中间量
    sin_zenith = np.sin(sub_zenith)
    cos_zenith = np.cos(sub_zenith)
    sin2 = sin_zenith ** 2
    sin4 = sin2 ** 2

    sqrt_term = np.sqrt(np.maximum(eta * eta - sin2, 0.0))

    # ======= 镜面偏振度 specular_rho =======
    denom_spec = (eta * eta - sin2) - eta * eta * sin2 + 2.0 * sin4
    specular_rho = np.zeros_like(denom_spec)
    mask_spec = (denom_spec != 0)
    specular_rho[mask_spec] = (
            2.0 * sin2[mask_spec] * cos_zenith[mask_spec] * sqrt_term[mask_spec]
            / denom_spec[mask_spec]
    )
    # 若 specular_rho 为轻微负数(>-1e-7), 置0
    small_neg_mask = (specular_rho < 1e-7) & (specular_rho > -1e-7)
    specular_rho[small_neg_mask] = 1e-3
    big_neg_mask = (specular_rho < -1e-7)
    # 你可选择 print 或截断:
    # specular_rho[big_neg_mask] = 0

    # ======= 漫反射偏振度 diffuse_rho =======
    numerator_diff = (eta - 1.0 / eta) ** 2 * sin2
    denom_diff = (2.0 + 2.0 * (eta * eta)
                  - (eta + 1.0 / eta) ** 2 * sin2
                  + 4.0 * cos_zenith * sqrt_term)
    diffuse_rho = np.zeros_like(denom_diff)
    mask_diff = (denom_diff != 0)
    diffuse_rho[mask_diff] = numerator_diff[mask_diff] / denom_diff[mask_diff]
    # 同理，轻微负数截断为0
    diff_neg_small = (diffuse_rho < 1e-7) & (diffuse_rho > -1e-7)
    diffuse_rho[diff_neg_small] = 1e-2
    # diff_neg_big = (diffuse_rho < -1e-7)

    # ======= 漫反射偏振角 diffuse_phi =======
    # sub_azimuth 已在 [0, pi]
    # 原逻辑: if azimuth>0 => azimuth else +pi
    # 由于已做了上面限制 => 这里可直接使用 sub_azimuth
    diffuse_phi = sub_azimuth

    # ======= 镜面偏振角 specular_phi =======
    specular_phi = np.where(sub_azimuth >= math.pi / 2,
                            sub_azimuth - math.pi / 2,
                            sub_azimuth + math.pi / 2)

    # 5) 写回输出数组
    DDoLP_img[1:-1, 1:-1] = diffuse_rho
    DAoLP_img[1:-1, 1:-1] = diffuse_phi
    SDoLP_img[1:-1, 1:-1] = specular_rho
    SAoLP_img[1:-1, 1:-1] = specular_phi

    return DDoLP_img, DAoLP_img, SDoLP_img, SAoLP_img, zenith_img

def generateReflectionByLabeledType(normal_img, binary_img, eta=1.5):
    """
    根据法向图和二值图像，直接利用二值图像决定每个像素使用 diffuse 还是 specular 参数，
    并生成最终的 DoLP、AoLP 以及反射类型图（reflect_img）。

    参数：
      normal_img: 法向图 (H, W, 3)
      binary_img: 已有的二值图像 (H, W)，0 表示 specular，1 表示 diffuse
      eta: 折射率(默认1.5)，传给 normal2DAoLP
      output_dir: 用于输出 reflect_vis_img_origin.png 的目录(默认 ../output)

    返回：
      DoLP_img, AoLP_img: 生成的偏振度和偏振角图像 (float32)
      reflect_img: 反射类型可视化图（0：specular，255：diffuse）(uint8)
    """
    # 1) 计算 diffuse & specular 对应的 DoLP/AoLP
    DDoLP_img, DAoLP_img, SDoLP_img, SAoLP_img, zenith_img = normal2DAoLP(normal_img, eta)

    # 2) 准备输出数组
    rows, cols = binary_img.shape
    DoLP_img = np.zeros((rows, cols), dtype=np.float32)
    AoLP_img = np.zeros((rows, cols), dtype=np.float32)
    reflect_img = np.zeros((rows, cols), dtype=np.uint8)

    # 3) 构造布尔掩码 (1=diffuse, 0=specular)
    type_img = binary_img.astype(np.uint8)
    mask_diffuse = (type_img == 1)
    mask_specular = (type_img == 0)

    # 4) 一次性填充 diffuse 区域
    DoLP_img[mask_diffuse] = DDoLP_img[mask_diffuse]
    AoLP_img[mask_diffuse] = DAoLP_img[mask_diffuse]
    reflect_img[mask_diffuse] = 255

    # 5) 一次性填充 specular 区域
    DoLP_img[mask_specular] = SDoLP_img[mask_specular]
    AoLP_img[mask_specular] = SAoLP_img[mask_specular]
    reflect_img[mask_specular] = 0

    return DoLP_img, AoLP_img, reflect_img