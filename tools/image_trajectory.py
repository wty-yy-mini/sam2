# -*- coding: utf-8 -*-
'''
@File    : image_trajectory.py
@Time    : 2026/01/08 20:32:07
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/posts/4856/
@Desc    : 在视频帧序列中叠加轨迹蒙版以生成频闪摄影效果图像。
'''
from PIL import Image
import numpy as np

base_dir = "./tools/VID_20251210_094125_frames_30fps"   # 视频裁剪主文件夹

origin_image_dir = f"{base_dir}/1_frame450-659"         # 原始图像文件夹
mask_dir = f"{base_dir}/1_frame450-659_masks"           # 分割掩码文件夹
base_image = f"{base_dir}/base_image.png"               # 基础背景图像, 后续在此基础上叠加蒙版
trajectory_idxs = [54, 72, 84, 94, 96, 109, 125, 133, 140, 152, 178, 192]  # 轨迹帧索引 (包含虚影和关键帧)

# 颜色定义
TAB_BLUE = np.array([31, 119, 180], dtype=np.uint8)
TAB_ORANGE = np.array([255, 127, 14], dtype=np.uint8)
TAB_GREEN = np.array([44, 160, 44], dtype=np.uint8)
TAB_RED = np.array([214, 39, 40], dtype=np.uint8)
TINT_STRENGTH = 0.6  # 颜色叠加强度

key_idx_color_map = {  # 关键帧索引到颜色的映射
    192: TAB_GREEN,  # Resumed Gait (恢复行走)
    133: TAB_RED,    # Impact Absorption (冲击吸收)
    109: TAB_BLUE,   # Rapid Adaptation (快速适应)
    96:  TAB_ORANGE, # Support Loss (支撑丢失)
    54:  TAB_GREEN,  # Steady-state Gait (稳态行走)
}

def get_img_and_key(idx):
    mask_path = f"{mask_dir}/{idx:05d}.jpg"
    origin_image_path = f"{origin_image_dir}/{idx:05d}.jpg"
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask > 200  # 边缘有些噪声, 加大mask阈值可以消去这些
    origin_image = np.array(Image.open(origin_image_path).convert("RGBA"))
    return origin_image, mask

result_image = np.array(Image.open(base_image).convert("RGBA"))

print("Processing trajectory frames (ghosts)...")
for i, traj_idx in enumerate(reversed(trajectory_idxs)):
    origin_image, mask = get_img_and_key(traj_idx)
    if traj_idx not in key_idx_color_map:
        result_image[mask] = origin_image[mask] * 0.4 + result_image[mask] * 0.6

print("Processing key frames with coloring...")
for key_idx in reversed(key_idx_color_map.keys()):
    origin_image, mask = get_img_and_key(key_idx)
    target_color = key_idx_color_map.get(key_idx)

    robot_pixels_rgba = origin_image[mask]
    robot_rgb = robot_pixels_rgba[:, :3]
    robot_alpha = robot_pixels_rgba[:, 3:4]
    color_overlay = np.full_like(robot_rgb, target_color)
    tinted_rgb = (robot_rgb * (1 - TINT_STRENGTH) + color_overlay * TINT_STRENGTH).astype(np.uint8)
    tinted_rgba = np.concatenate([tinted_rgb, robot_alpha], axis=1)
    result_image[mask] = tinted_rgba

result_image = Image.fromarray(result_image)
result_image.save(f"{base_dir}/result_image.png")
print(f"Result image saved to {base_dir}/result_image.png")
