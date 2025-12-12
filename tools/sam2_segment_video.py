"""
# Extract frames from video using ffmpeg
python tools/extract_frames_from_video.py  # change video path and output folder inside the script
python tools/sam2_segment_video_extract.py \
    --video-parent-dir /home/yy/Downloads/VID_20251210_094125_frames_30fps \
    --show-prompts \
    --save-mask-video \
    --save-mask-frames \
    --mask-color avg
"""
from pathlib import Path

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

import json
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from typing import Literal

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
    def __init__(self,
            show_prompts=False,
            save_mask_video=False,
            save_mask_frames=False,
            mask_color: Literal['white', 'avg']="white",
        ):
        self.show_prompts = show_prompts
        self.save_mask_video = save_mask_video
        self.save_mask_frames = save_mask_frames
        self.mask_color = mask_color
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
            plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100)
            plt.title(f"Frame {frame_idx} with Box Prompts")
            plt.imshow(Image.open(self.frames[frame_idx]))
            add_infos = {}
            for shape in shapes:
                if shape['shape_type'] not in ['rectangle', 'point']:
                    continue
                if shape['shape_type'] == 'rectangle':
                    label = shape['label']
                elif shape['shape_type'] == 'point':
                    label_with_flag = shape['label']
                    label = label_with_flag.split('_')[0]
                    point_label = 0 if 'neg' in label_with_flag else 1
                if label not in self.label2obj_id:
                    self.label2obj_id[label] = len(self.label2obj_id)
                if self.label2obj_id[label] not in add_infos:
                    add_infos[self.label2obj_id[label]] = {
                        'inference_state': self.inference_state,
                        'frame_idx': frame_idx,
                        'obj_id': self.label2obj_id[label],
                        'box': [],
                        'points': [],
                        'labels': []
                    }
                infos = add_infos[self.label2obj_id[label]]

                if shape['shape_type'] == 'rectangle':
                    box = shape['points']  # [[x0, y0], [x1, y1]]
                    x0, y0 = box[0]
                    x1, y1 = box[1]
                    box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                    infos['box'].append(box)
                    show_box(box, plt.gca())
                if shape['shape_type'] == 'point':
                    point = shape['points']  # [[x, y]]
                    infos['points'].append(point[0])
                    infos['labels'].append(point_label)
                    show_points(np.array(point), np.array([point_label]), plt.gca())

            if len(add_infos) != 0:
                for infos in add_infos.values():
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(**infos)

            for i, out_obj_id in enumerate(out_obj_ids):
                show_mask((out_mask_logits[i] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
            if self.show_prompts:
                plt.tight_layout()
                plt.axis('off')
                plt.show()
            else:
                plt.close()
            self.num_prompts += 1
    
    def segment_frames(self):
        if self.save_mask_frames:
            output_dir  = self.video_dir.parent / f"{self.video_dir.name}_masks"
            output_dir.mkdir(exist_ok=True, parents=True)
        if self.save_mask_video:
            output_video = self.video_dir.parent / f"{self.video_dir.name}_masked.mp4"
            writer = imageio.get_writer(output_video, fps=30)
        video_segments = {}
        if self.num_prompts > 0:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        for frame_idx in tqdm(range(len(self.frames))):
            # Write video
            img = Image.open(self.frames[frame_idx])
            img = np.array(img)
            if self.save_mask_video and frame_idx not in video_segments:
                writer.append_data(img)
                continue
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                mask = out_mask.reshape(self.h, self.w)
                pixels = img[mask].astype(np.float32)
                if len(pixels) > 0:
                    if self.mask_color == 'avg':
                        img[mask] = np.mean(pixels, axis=0).astype(np.uint8)  # color the masked area with mean color
                    elif self.mask_color == 'white':
                        img[mask] = 255  # white out the masked area

            if self.save_mask_video:
                writer.append_data(img)

            # Save segmented frames
            if self.save_mask_frames:
                mask = np.zeros_like(img)
                for out_obj_id, out_mask in video_segments[frame_idx].items():
                    mask[out_mask.reshape(self.h, self.w)] = 255
                output_path = output_dir / f"{frame_idx:05d}.jpg"
                Image.fromarray(mask).save(output_path)

            # Matplotlib visualization (optional)
            # plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100)
            # plt.title(f"Frame {frame_idx}")
            # plt.imshow(Image.open(self.frames[frame_idx]))
            # for out_obj_id, out_mask in video_segments[frame_idx].items():
            #     show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            # plt.axis('off')
            # output_path = output_dir / f"{frame_idx:05d}.jpg"
            # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            # plt.close()
        writer.close()
        print(f"Segmented video saved to: {output_video}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-parent-dir", type=str, default="", help="Path to the parent of video frames directory.")
    parser.add_argument("--show-prompts", action="store_true", help="Whether to show prompts visualization.")
    parser.add_argument("--save-mask-video", action="store_true", help="Whether to save masked video.")
    parser.add_argument("--save-mask-frames", action="store_true", help="Whether to save masked frames.")
    parser.add_argument("--mask-color", type=str, default="white", help="Color to use for masking (e.g., 'white', 'avg').")
    args = parser.parse_args()

    show_prompts = args.show_prompts
    video_parent_dir = args.video_parent_dir
    save_mask_video = args.save_mask_video
    save_mask_frames = args.save_mask_frames
    mask_color = args.mask_color
    kwargs = {
        "show_prompts": show_prompts,
        "save_mask_video": save_mask_video,
        "save_mask_frames": save_mask_frames,
        "mask_color": mask_color,
    }

    video_dirs = [x for x in sorted(Path(video_parent_dir).glob("*")) if x.is_dir()]
    for video_dir in video_dirs:
        idx = int(video_dir.name.split("_")[0])
        if 'masks' == video_dir.name.split('_')[-1]:
            continue
        if idx >= 1:
        # if 2 <= idx <= 9 and idx not in []:
            sam2_segment_video_processor = SAM2SegmentVideoProcessor(**kwargs)
            print(f"Processing video directory: {video_dir}")
            sam2_segment_video_processor.init_state(str(video_dir))
            sam2_segment_video_processor.load_frame_prompt()
            sam2_segment_video_processor.segment_frames()
