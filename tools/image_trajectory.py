from PIL import Image
import numpy as np

base_dir = "/home/yy/Downloads/VID_20251210_094125_frames_30fps"

origin_image_dir = f"{base_dir}/1_frame450-659"
mask_dir = f"{base_dir}/1_frame450-659_masks"
base_image = f"{base_dir}/base_image.png"
trajectory_idxs = [54, 72, 84, 94, 96, 109, 125, 133, 140, 152, 178, 192]
key_idxs = [54, 96, 109, 133, 192]

TAB_BLUE = np.array([31, 119, 180], dtype=np.uint8)
TAB_ORANGE = np.array([255, 127, 14], dtype=np.uint8)
TAB_GREEN = np.array([44, 160, 44], dtype=np.uint8)
TAB_RED = np.array([214, 39, 40], dtype=np.uint8)
TINT_STRENGTH = 0.6 

key_idx_color_map = {
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
    # if i == 0:  # 保存第一个图片, 作为base image基础, ps消去无用的背景
    #     result_image = origin_image.copy()
    #     Image.fromarray(result_image).save("/home/yy/Downloads/VID_20251210_094125_frames_30fps/base_image.jpg")

    if traj_idx not in key_idxs:
        result_image[mask] = origin_image[mask] * 0.4 + result_image[mask] * 0.6
        # result_image[mask, 3] = 255  # 保持不透明度为255

print("Processing key frames with coloring...")
for key_idx in reversed(key_idxs):
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
