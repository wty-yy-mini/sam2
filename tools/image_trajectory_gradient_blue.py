# -*- coding: utf-8 -*-
'''
@File    : image_trajectory.py
@Time    : 2026/01/08 20:32:07
@Author  : wty-yy
@Version : 2.0
@Blog    : https://wty-yy.github.io/posts/4856/
@Desc    : 在视频帧序列中叠加轨迹蒙版以生成频闪摄影效果图像，
根据收尾关键帧从LCH空间中自动计算蓝色调渐变色。
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
TINT_STRENGTH = 0.6  # 颜色叠加强度

def get_auto_blue_gradient(idx, start_idx, end_idx):
    # 1. 归一化进度 t [0, 1]
    t = np.clip((idx - start_idx) / (end_idx - start_idx), 0, 1)
    
    # 2. 直接在 LCH 空间定义演化路径
    # 亮度 L 从 90 (浅) 降到 30 (深)
    res_l = 90 - t * 60 
    
    # 饱和度 C 从 10 (灰) 升到 60 (纯净蓝色)
    res_c = 10 + t * 50 
    
    # 色相 H 保持在蓝色区间 (约 240-260 度，弧度表示)
    res_h = 250 * (np.pi / 180) 
    
    # 3. 转回 RGB
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
    
    target_color = get_auto_blue_gradient(
        key_idx, start_idx, end_idx
    )

    robot_pixels_rgba = origin_image[mask]
    robot_rgb = robot_pixels_rgba[:, :3]
    robot_alpha = robot_pixels_rgba[:, 3:4]
    
    color_overlay = np.full_like(robot_rgb, target_color)
    tinted_rgb = (robot_rgb * (1 - TINT_STRENGTH) + color_overlay * TINT_STRENGTH).astype(np.uint8)
    tinted_rgba = np.concatenate([tinted_rgb, robot_alpha], axis=1)
    result_image[mask] = tinted_rgba

result_image = Image.fromarray(result_image)
result_image.save(f"{base_dir}/blue_result_image.png")
print(f"Result image saved to {base_dir}/blue_result_image.png")
