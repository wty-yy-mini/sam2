from PIL import Image
import numpy as np

origin_image_dir = "/home/yy/Downloads/VID_20251210_094125_frames_30fps/1_frame450-659"
mask_dir = "/home/yy/Downloads/VID_20251210_094125_frames_30fps/1_frame450-659_masks"
trajectory_idxs = [54, 72, 84, 95, 100, 105, 109, 125, 128, 130, 133, 137, 141, 162, 182, 192]

for traj_idx in trajectory_idxs:
    mask_path = f"{mask_dir}/{traj_idx:05d}.jpg"
    origin_image_path = f"{origin_image_dir}/{traj_idx:05d}.jpg"

    mask = Image.open(mask_path).convert("L")
    origin_image = Image.open(origin_image_path).convert("RGB")

    mask = np.array(mask)
    origin_image = np.array(origin_image)

    origin_image[mask > 0] = [255, 0, 0]  # Highlight the mask area in red

    result_image = Image.fromarray(origin_image)
    result_image.show()
