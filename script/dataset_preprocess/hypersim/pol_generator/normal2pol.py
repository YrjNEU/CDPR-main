import cv2
import numpy as np
import math
import os
import h5py
from pol_generator.utils import *  # 这里包含 setNormal, generateReflectionByLabeledType, dolp_aolp_to_polars, PolarsToDolpAolp 等
from pylab import count_nonzero, clip, np
# 全局参数
eta = 1.5
def tone_map(rgb, entity_id_map):
    assert (entity_id_map != 0).all()

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = (
        90  # we want this percentile brightness value in the unmodified image...
    )
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = entity_id_map != -1

    if count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        brightness = (
            0.3 * rgb[:, :, 0] + 0.59 * rgb[:, :, 1] + 0.11 * rgb[:, :, 2]
        )  # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness[valid_mask]

        eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = (
                np.power(brightness_nth_percentile_desired, inv_gamma)
                / brightness_nth_percentile_current
            )

    rgb_color_tm = np.power(np.maximum(scale * rgb, 0), gamma)
    rgb_color_tm = clip(rgb_color_tm, 0, 1)
    return rgb_color_tm


def readhdf5(filepath):
    """
    假设标签文件为文本格式，按空白字符分隔整数。
    """
    with h5py.File(filepath, "r") as f:
        label_mat = f['dataset'][:]
    return label_mat

def write_hdf5(filepath, data):
    """
    将 data 数组写入到 HDF5 文件中
    """
    label_mat = 'dataset'
    data = np.asarray(data)
    print(data.shape)
    with h5py.File(filepath, 'w') as f_out:
        f_out.create_dataset(label_mat, data=data, dtype=data.dtype)
    print(f'[Info] HDF5 mask saved to {filepath}')


def generate_single_syn_daolp_by_labeled_type(rgb_img, normal_img, binary_img):
    """
    输入:
        rgb_img: 读入的RGB图 (H*W*3)
        normal_img: 读入并处理后的法向图 (H*W*3)
        binary_img: 读入的镜面/非镜面二值图 (H*W)

    返回:
        dp_st: 归一化后的DoLP图(0~255, uint8)
        ap_st: 归一化后的AoLP图(0~255, uint8)
    """
    # 先根据给定法向与镜面掩码生成理论 DoLP, AoLP, 以及镜面反射比
    DoLP_img, AoLP_img, reflect_img = generateReflectionByLabeledType(normal_img, binary_img)

    # 将 DoLP 与 AoLP 转换为 4 张偏振图（0°, 45°, 90°, 135°）
    polars_img = dolp_aolp_to_polars(DoLP_img, AoLP_img, rgb_img)

    # 为每张偏振图添加噪声并修改像素值
    for i in range(4):
        polar_img = polars_img[i]
        # 将每张偏振图像素值乘以 300，再转换为 float32
        for j in range(3):
            polar_img[:, :, j] = polar_img[:, :, j].astype(np.float32)
            polar_img[:, :, j] = np.rint(polar_img[:, :, j]).astype(np.float32)
        # 添加噪声（均值 0，标准差 0.5）
        noise = np.random.normal(0, 3, polar_img.shape).astype(polar_img.dtype)
        polars_img[i] = cv2.subtract(polar_img, noise)

    # 由4张偏振图逆推DoLP与AoLP（含噪声）
    DoLP_img_clip, AoLP_img_clip = PolarsToDolpAolp(polars_img)

    # 将结果转换到 0~255 的 uint8 区间
    dp_st = np.rint(DoLP_img_clip * 255).astype(np.uint8)
    ap_st = np.rint((AoLP_img_clip / math.pi) * 255).astype(np.uint8)

    return dp_st, ap_st


def main():
    input_root = '/media/neu/YINGJIE/hypersim/data/ai_001_001/images'
    output_root = '/media/neu/YINGJIE/hypersim/data/ai_001_001/images/output_temp'

    # 读入数据
    rgb_path = os.path.join(input_root, "scene_cam_00_final_hdf5/frame.0000.color.hdf5")
    norm_path = os.path.join(input_root, "scene_cam_00_geometry_hdf5/frame.0000.normal_cam.hdf5")
    render_entity_id_path = os.path.join(input_root, "scene_cam_00_geometry_hdf5/frame.0000.render_entity_id.hdf5")
    label_path = os.path.join(output_root, 'frame.0000.mirror_mask.hdf5')

    rgb_img = readhdf5(rgb_path)
    render_entity_id = readhdf5(render_entity_id_path)
    if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
        # 由于一些通道顺序原因，可能需要做 (R, G, B) -> (B, G, R) 的转换
        rgb_color_tm = tone_map(rgb_img, render_entity_id)
        rgb_img = (rgb_color_tm * 255).astype(np.uint8)  # [H, W, RGB]
        rgb_img = rgb_img[..., ::-1]

    normal_gt_img = readhdf5(norm_path)
    binary_img = readhdf5(label_path)

    if normal_gt_img is None or rgb_img is None or binary_img is None:
        print("[Error] 加载图像失败，请检查路径")
        return

    # 对法向图做预处理
    normal_img = normal_gt_img.copy()
    normal_img = setNormal(normal_img)

    # 调用核心计算函数，得到可视化的 DoLP 和 AoLP（uint8）
    dp_st, ap_st = generate_single_syn_daolp_by_labeled_type(rgb_img, normal_img, binary_img)

    # 将结果写出到文件
    output_path_dp = os.path.join(output_root, "dpol.png")
    output_path_ap = os.path.join(output_root, "apol.png")
    cv2.imwrite(output_path_dp, dp_st)
    cv2.imwrite(output_path_ap, ap_st)

    # 若需要将浮点数形式的 DoLP/AoLP 写入 hdf5，可根据需要自行组装，这里演示写 AoLP_img_clip
    # 注意此时若要写 hdf5，需要在 generate_single_syn_daolp_by_labeled_type 里修改逻辑，
    # 同时把浮点形式的 AoLP_img_clip 也返回，这里只演示写 ap_st。
    hdf5_path_ap = os.path.join(output_root, "apol.hdf5")
    write_hdf5(hdf5_path_ap, ap_st)

    print("处理完成。")

if __name__ == "__main__":
    main()
