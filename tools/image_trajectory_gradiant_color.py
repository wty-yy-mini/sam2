# -*- coding: utf-8 -*-
'''
@File    : image_trajectory.py
@Time    : 2026/01/08 20:32:07
@Author  : wty-yy
@Version : 2.0
@Blog    : https://wty-yy.github.io/posts/4856/
@Desc    : 在视频帧序列中叠加轨迹蒙版以生成频闪摄影效果图像，
根据收尾关键帧从LCH空间中自动计算渐变色。
pip install scikit-image
'''
from PIL import Image
import numpy as np
from skimage import color

base_dir = "./tools/VID_20251210_094125_frames_30fps"   # 视频裁剪主文件夹

origin_image_dir = f"{base_dir}/1_frame450-659"         # 原始图像文件夹
mask_dir = f"{base_dir}/1_frame450-659_masks"           # 分割掩码文件夹
base_image = f"{base_dir}/base_image.png"               # 基础背景图像, 后续在此基础上叠加蒙版
trajectory_idxs = [54, 72, 84, 94, 96, 109, 125, 133, 140, 152, 178, 192]  # 轨迹帧索引 (包含虚影和关键帧)
start_idx = trajectory_idxs[0]  # 颜色渐变起点索引
end_idx = trajectory_idxs[-1]   # 颜色渐变终点索引

# 颜色定义
TAB_BLUE = np.array([31, 119, 180], dtype=np.uint8)   # 开始
TAB_ORANGE = np.array([255, 127, 14], dtype=np.uint8)
TAB_GREEN = np.array([44, 160, 44], dtype=np.uint8)
TAB_RED = np.array([214, 39, 40], dtype=np.uint8)     # 结束
TINT_STRENGTH = 0.6  # 颜色叠加强度
START_COLOR = TAB_BLUE  # 起点颜色
END_COLOR = TAB_RED     # 终点颜色

def get_auto_spectral_color(idx, start_idx, end_idx, start_rgb, end_rgb):
    """
    基于首尾颜色，在 LCH 空间自动插值产生光谱渐变。
    """
    # 1. 归一化进度 t [0, 1]
    t = np.clip((idx - start_idx) / (end_idx - start_idx), 0, 1)
    
    # 2. 将首尾 RGB 转换为 Lab，再转为 LCH
    def rgb_to_lch(rgb):
        lab = color.rgb2lab(rgb.reshape(1, 1, 3) / 255.0)
        # LCH 并不是 skimage 直接提供的，我们可以手动从 Lab 转极坐标
        # 或者直接使用感知更均匀的 HSV，但 LCH 对论文制图更严谨
        l = lab[0, 0, 0]
        a, b = lab[0, 0, 1], lab[0, 0, 2]
        c = np.sqrt(a**2 + b**2)
        h = np.arctan2(b, a) # 弧度
        return np.array([l, c, h])

    lch_start = rgb_to_lch(start_rgb)
    lch_end = rgb_to_lch(end_rgb)
    
    # 3. 在 LCH 空间线性插值
    base_l = lch_start[0] + t * (lch_end[0] - lch_start[0])
    
    # --- 引入亮度增强 (Brightness Boost) ---
    # 使用 t*(1-t) 构造一个在 t=0.5 处达到峰值的抛物线
    # 这里的 20 代表亮度增益强度，可以根据需要调整 (建议范围 20-40)
    l_boost = 20 * (4 * t * (1 - t)) 
    res_l = base_l + l_boost

    res_c = lch_start[1] + t * (lch_end[1] - lch_start[1])
    
    # 确保亮度不溢出 (0-100)
    res_l = np.clip(res_l, 0, 100)

    # 处理色相环绕：确保是从蓝->绿->黄->红的路径
    h_start, h_end = lch_start[2], lch_end[2]
    if h_end > h_start: h_end -= 2 * np.pi # 确保逆时针旋转经过青/绿
    res_h = h_start + t * (h_end - h_start)
    
    # 4. 转回 Lab 再转回 RGB
    res_a = res_c * np.cos(res_h)
    res_b = res_c * np.sin(res_h)
    res_lab = np.array([res_l, res_a, res_b])
    
    rgb_res = color.lab2rgb(res_lab.reshape(1, 1, 3)).flatten()
    return (rgb_res * 255).astype(np.uint8)

def get_img_and_key(idx):
    mask_path = f"{mask_dir}/{idx:05d}.jpg"
    origin_image_path = f"{origin_image_dir}/{idx:05d}.jpg"
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask > 200  # 边缘有些噪声, 加大mask阈值可以消去这些
    origin_image = np.array(Image.open(origin_image_path).convert("RGBA"))
    return origin_image, mask

result_image = np.array(Image.open(base_image).convert("RGBA"))

print("Processing key frames with automatic spectral coloring...")
for key_idx in reversed(trajectory_idxs):
    origin_image, mask = get_img_and_key(key_idx)
    
    target_color = get_auto_spectral_color(
        key_idx, start_idx, end_idx, START_COLOR, END_COLOR
    )

    robot_pixels_rgba = origin_image[mask]
    robot_rgb = robot_pixels_rgba[:, :3]
    robot_alpha = robot_pixels_rgba[:, 3:4]
    
    color_overlay = np.full_like(robot_rgb, target_color)
    tinted_rgb = (robot_rgb * (1 - TINT_STRENGTH) + color_overlay * TINT_STRENGTH).astype(np.uint8)
    tinted_rgba = np.concatenate([tinted_rgb, robot_alpha], axis=1)
    result_image[mask] = tinted_rgba

result_image = Image.fromarray(result_image)
result_image.save(f"{base_dir}/color_result_image.png")
print(f"Result image saved to {base_dir}/color_result_image.png")
