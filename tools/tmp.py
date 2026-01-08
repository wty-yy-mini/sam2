"""
# Extract frames from video using ffmpeg
python tools/extract_frames_from_video.py  # change video path and output folder inside the script
"""
from pathlib import Path

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from PIL import Image
import cv2

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

class SAM2SegmentVideoProcessor:
    def __init__(self):
        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        from sam2.build_sam import build_sam2_video_predictor

        sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        # sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
        # model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        self.label2obj_id = {}
        self.num_prompts = 0

    def init_state(self, video_dir: str):
        self.video_dir = Path(video_dir)
        self.frames = sorted(Path(video_dir).rglob("*.jpg"))
        self.w, self.h = Image.open(self.frames[0]).size
        print(f"found {len(self.frames)} frames")

        # Init model
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(self.inference_state)

    def load_frame_prompt(self):
        """ Find all json files for frame prompts (from LabelMe) """
        for frame_json in sorted(self.video_dir.rglob("*.json")):
            frame_idx = int(frame_json.stem)
            with open(frame_json, "r") as f:
                labelme_data = json.load(f)
            shapes = labelme_data["shapes"]
            if len(shapes) == 0: continue
            # plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100)
            # plt.title(f"Frame {frame_idx} with Box Prompts")
            # plt.imshow(Image.open(self.frames[frame_idx]))
            for shape in shapes:
                if shape['shape_type'] != 'rectangle': continue
                label = shape['label']
                if label not in self.label2obj_id:
                    self.label2obj_id[label] = len(self.label2obj_id)
                box = shape['points']  # [[x0, y0], [x1, y1]]
                x0, y0 = box[0]
                x1, y1 = box[1]
                box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=self.label2obj_id[label],
                    box=box
                )
                # show_box(box, plt.gca())
            for i, out_obj_id in enumerate(out_obj_ids):
                show_mask((out_mask_logits[i] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
            # plt.axis('off')
            # plt.show()
            self.num_prompts += 1

    def segment_frames(self):
        # output_dir  = self.video_dir.parent / f"{self.video_dir.name}_segmented"
        # output_dir.mkdir(exist_ok=True, parents=True)
        output_video = self.video_dir.parent / f"{self.video_dir.name}_segmented.avi"
        video_segments = {}
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'XVID'), fps=30, frameSize=(self.w, self.h))
        if self.num_prompts > 0:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        for frame_idx in tqdm(range(len(self.frames))):
            # Write video
            if frame_idx not in video_segments:
                img = cv2.imread(self.frames[frame_idx])
                writer.write(img)
                continue
            img = cv2.imread(self.frames[frame_idx])
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                mask = out_mask.reshape(self.h, self.w)
                pixels = img[mask].astype(np.float32)
                if len(pixels) > 0:
                    # img[mask] = np.mean(pixels, axis=0).astype(np.uint8)  # (Optional 1) color the masked area with mean color
                    img[mask] = 255  # (Optional 2) white out the masked area

            writer.write(img)

            # Save segmented frames
            # img = Image.open(self.frames[frame_idx])
            # img = np.array(img)
            # for out_obj_id, out_mask in video_segments[frame_idx].items():
            #     img[out_mask.reshape(self.h, self.w)] = 255
            # output_path = output_dir / f"{frame_idx:05d}.png"
            # Image.fromarray(img).save(output_path)

            # Matplotlib visualization (optional)
            # plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100)
            # plt.title(f"Frame {frame_idx}")
            # plt.imshow(Image.open(self.frames[frame_idx]))
            # for out_obj_id, out_mask in video_segments[frame_idx].items():
            #     show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            # plt.axis('off')
            # output_path = output_dir / f"{frame_idx:05d}.png"
            # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            # plt.close()
        writer.release()
        print(f"Segmented video saved to: {output_video}")

if __name__ == '__main__':
    video_parent_dir = "/home/yy/Videos/sam2_mask_demo/g1_dance_demo_frames_30fps"
    video_dirs = [x for x in sorted(Path(video_parent_dir).glob("*")) if x.is_dir()]
    for video_dir in video_dirs:
        idx = int(video_dir.name.split("_")[0])
        if idx >= 1:
        # if 2 <= idx <= 9 and idx not in []:
            sam2_segment_video_processor = SAM2SegmentVideoProcessor()
            print(f"Processing video directory: {video_dir}")
            sam2_segment_video_processor.init_state(str(video_dir))
            sam2_segment_video_processor.load_frame_prompt()
            sam2_segment_video_processor.segment_frames()
