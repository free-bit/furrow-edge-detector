#!/usr/bin/env python

import json
import os

import numpy as np
from PIL import Image
# from scipy.ndimage import shift, rotate
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from utils.helpers import coord_to_mask, take_items

DEPTH_EXT = "_depth.npy"	
EDGE_EXT = "_edge_pts.npy"	
RGB_EXT = "_rgb.png"	
DRGB_EXT = "_depth.png"
TIME_EXT = "_time.json"
MAX_DEPTH = 65.53500311274547

T_MAP = {
    "affine": F.affine,
    "center_crop": T.Lambda(lambda x:  F.center_crop(x, output_size=[400,400])),
    "crop_down": T.Lambda(lambda x: F.crop(x, top=80, left=120, height=400, width=400)),
    "gaussian_blur": F.gaussian_blur,
    "normalize_imagenet": {
        "1C": T.Lambda(lambda x: F.normalize(x, mean=0.449, std=0.226)),
        "3C": T.Lambda(lambda x: F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    },
    "normalize_furrowset": T.Lambda(lambda x: F.normalize(x, mean=0.5, std=0.5)), # Pixel values in range: [-1,1] # TODO: Compute mean & variance in the dataset
    "resize": F.resize,
    "rotate": F.rotate,
}

class FurrowDataset(Dataset):

    def __init__(self, data_args):
        self.data_args = data_args
        self.validate_data_path()

        self.folder_id = None
        self.frame_ids = []
        self.tags = []
        self.timestamps = {}
        self.size = 0
        
        start = data_args.get("start", 0)
        end = data_args.get("end", np.inf)
        max_frames = data_args.get("max_frames", np.inf)
        step = data_args.get("step", 1)
        self.read_frame_metadata()
        self.frame_count = len(self.frame_ids)
        self.take_frames(start, end, max_frames, step)

    def get_args(self):
        return self.data_args

    def save_args(self, save_path):
        if not os.path.splitext(save_path)[-1]:
            save_path = save_path + ".json"

        with open(save_path, "w") as file:
            json.dump(self.data_args, file, indent=4)

    def validate_data_path(self):
        data_path = self.data_args['data_path']
        files = os.listdir(data_path)
        for file in files:
            file_path = os.path.join(data_path, file)
            assert os.path.isfile(file_path), f"{file_path} is not a file!"
            assert file.endswith(("depth.npy", "pts.npy", "rgb.png", "depth.png", "vis.png", "time.json")),\
                f"{file} has an unknown extension!"

    def read_frame_metadata(self):
        """Read file ids and timestamps under a single-folder dataset containing frames."""
        data_path = self.data_args['data_path']
        load_time = self.data_args.get("load_time", False)
        self.folder_id = os.path.basename(data_path)
        files = os.listdir(data_path)
        
        # Build an index based on RGB data (assuming always exists) for frames available
        frame_ids = []
        augs = []
        for file in files:
            if file.endswith(RGB_EXT):
                metadata = file.split("_")
                frame_ids.append(int(metadata[0])) # ID of the frame
                aug_lst = metadata[1:-1]
                if aug_lst:
                    augs.append(aug_lst)           # Augmentation tag for the frame (if applied)

        # Sort the index based on frame id
        sort_indices = np.argsort(frame_ids)
        self.frame_ids = [frame_ids[i] for i in sort_indices]
        self.augs = [augs[i] for i in sort_indices] if augs else []
        
        if load_time:
            file = str(self.folder_id) + TIME_EXT
            path = os.path.join(data_path, file)
            with open(path) as f:
                self.timestamps = json.load(f)

    def take_frames(self, start=0, end=np.inf, max_frames=np.inf, step=1):
        """Shrink the number of frames to read according to given range and count"""
        self.frame_ids = take_items(self.frame_ids, start, end, max_frames, step)
        self.augs = take_items(np.array(self.augs, dtype=object), start, end, max_frames, step)

    def __len__(self):
        # Size <-> Number of frames
        return len(self.frame_ids)

    def get_frame_files(self, idx, load_darr, load_rgb, load_drgb, load_edge, load_time):
        # General purpose method to load data only
        data_path = self.data_args['data_path']
        allow_missing_files = self.data_args.get('allow_missing_files', False)
        
        frame_id = self.frame_ids[idx]
        augs, tag = [], ''
        if self.augs:
            augs = self.augs[idx]
            tag = "_" + "_".join(augs)

        frame_files = {
            "frame_id": frame_id,
            "augs": augs
        }

        # Load edge mask
        if load_edge:
            edge_file = str(frame_id) + tag + EDGE_EXT
            edge_file = os.path.join(data_path, edge_file)
            try:
                frame_files['edge_pixels'] = np.load(edge_file) # np.array: np.int64
            except FileNotFoundError:
                frame_files['edge_pixels'] = []
                if not allow_missing_files:
                    print(f"{edge_file} does not exist. Use allow_missing_files=True to allow empty edge masks.")
                    exit(-1)

        # Load depth as array only
        if load_darr:
            darr_file = str(frame_id) + tag + DEPTH_EXT
            darr_file = os.path.join(data_path, darr_file)
            frame_files['depth_arr'] = np.load(darr_file)       # np.array: np.float64

        # Load RGB image only
        if load_rgb:
            rgb_file = str(frame_id) + tag + RGB_EXT
            rgb_file = os.path.join(data_path, rgb_file)
            frame_files['rgb_img'] = Image.open(rgb_file)       # PIL.Image: np.uint8
        
        # Load depth as image only
        if load_drgb:
            drgb_file = str(frame_id) + tag + DRGB_EXT
            drgb_file = os.path.join(data_path, drgb_file)
            frame_files['depth_img'] = Image.open(drgb_file)    # PIL.Image: np.uint8

        # Load timestamps
        # if load_time:
        #     frame_files['time'] = self.timestamps[str(frame_id)]
        
        return frame_files
        
    def __getitem__(self, idx):
        # torch.utils.data.DataLoader class specific method: 
        # 1) Loads data with get_frame_files
        # 2) Adjusts input channels (if necessary)
        # 3) Applies torch transforms

        # Loading data as specified:
        input_format = self.data_args.get("input_format", 'darr')
        load_edge = self.data_args.get("load_edge", True)
        load_darr = input_format in ('darr', 'rgb-darr')
        load_rgb = input_format in ('rgb', 'rgb-darr', 'rgb-drgb')
        load_drgb = input_format in ('drgb', 'rgb-drgb')
        load_time = self.data_args.get("load_time", False)
        shape = self.data_args.get("shape", (480, 640))
        edge_width = self.data_args['edge_width']

        input_trans = [F.to_tensor]  # Pixel values in range: [0,1]
        target_trans = [F.to_tensor] # Pixel values in range: [0,1]
        crop_down = self.data_args.get('crop_down', True)
        normalize = self.data_args.get('normalize', True)
        if crop_down:
            input_trans.append(T_MAP['crop_down'])
            target_trans.append(T_MAP['crop_down'])
        
        frame_files = self.get_frame_files(idx, load_darr, load_rgb, load_drgb, load_edge, load_time)
        sample = {
            # "frame_id": frame_files["frame_id"]
        }
        
        # Target: edge_pixels -> edge_mask -> transform (if loaded)
        if load_edge:
            edge_pixels = frame_files['edge_pixels']
            edge_mask = coord_to_mask(shape, edge_pixels, thickness=edge_width) # np.uint8
            
            target_trans = T.Compose(target_trans)
            sample['target'] = target_trans(edge_mask)
        
        # Input Option-1: depth_arr: (C:1) -> (C:3) -> transform (if loaded)
        if input_format == "darr":
            depth_arr = frame_files['depth_arr']
            depth_arr = np.rint(255 * (depth_arr / MAX_DEPTH))               # Expand range from [0, MAX_DEPTH] to [0, 255]
            depth_arr = depth_arr.astype(np.uint8)                           # np.float64 -> np.uint8
            depth_arr = np.stack([depth_arr, depth_arr, depth_arr], axis=-1) # (C:1) -> (C:3)
            
            if normalize:
                input_trans.append(T_MAP['normalize_imagenet']['3C'])
            input_trans = T.Compose(input_trans)
            sample['input'] = input_trans(depth_arr) # TODO: This might be made optional

        # Input Option-2: rgb_img: (C:3) -> transform (if loaded)
        elif input_format == "rgb":
            if normalize:
                input_trans.append(T_MAP['normalize_imagenet']['3C'])
            input_trans = T.Compose(input_trans)
            sample['input'] = input_trans(frame_files['rgb_img'])
        
        # Input Option-3: depth_img: (C:3) -> transform (if loaded)
        elif input_format == "drgb":
            if normalize:
                input_trans.append(T_MAP['normalize_imagenet']['3C'])
            input_trans = T.Compose(input_trans)
            sample['input'] = input_trans(frame_files['depth_img'])
        
        # Input Option-4: rgb_img + depth_arr: (C:3) + (C:1) -> (C:4) -> transform (if loaded)
        elif input_format == "rgb-darr":
            rgb_trans = input_trans[:]
            darr_trans = input_trans[:]
            if normalize:
                rgb_trans.append(T_MAP['normalize_imagenet']['3C'])
                darr_trans.append(T_MAP['normalize_imagenet']['1C'])
            rgb_trans = T.Compose(rgb_trans)
            darr_trans = T.Compose(darr_trans)

            rgb_img = frame_files['rgb_img']
            rgb_img = rgb_trans(rgb_img)

            depth_arr = frame_files['depth_arr']
            depth_arr = np.rint(255 * (depth_arr / MAX_DEPTH))               # Expand range from [0, MAX_DEPTH] to [0, 255]
            depth_arr = depth_arr.astype(np.uint8)                           # np.float64 -> np.uint8
            depth_arr = darr_trans(depth_arr)
            
            sample['input'] = torch.cat((rgb_img, depth_arr), dim=0)
        
        # Input Option-5: rgb_img + depth_img: (C:3) + (C:3) -> (C:6) -> transform (if loaded)
        elif input_format == "rgb-drgb":
            if normalize:
                input_trans.append(T_MAP['normalize_imagenet']['3C'])
            input_trans = T.Compose(input_trans)

            rgb_img = input_trans(frame_files['rgb_img'])
            depth_img = input_trans(frame_files['depth_img'])
            sample['input'] = torch.cat((rgb_img, depth_img), dim=0)

        else:
            raise NotImplementedError
        
        return sample

    def __str__(self):
        data_path = self.data_args['data_path']
        input_format = self.data_args.get("input_format", None)
        load_darr = input_format in ('darr', 'rgb-darr')
        load_rgb = input_format in ('rgb', 'rgb-darr', 'rgb-drgb')
        load_drgb = input_format in ('drgb', 'rgb-drgb')
        load_edge = self.data_args.get("load_edge", True)
        load_time = self.data_args.get("load_time", False)
        
        info = f"Status of {self.folder_id} dataset:\n"+\
               f"* Number of frames loaded/total: {len(self.frame_ids)}/{self.frame_count}\n"+\
               f"* Input format: {input_format}\n"+\
               f"* Load depth as array: {load_darr}\n"+\
               f"* Load RGB: {load_rgb}\n"+\
               f"* Load depth as RGB: {load_drgb}\n"+\
               f"* Load edge coordinates (ground truth): {load_edge}\n"+\
               f"* Load timestamps: {load_time}\n"+\
               f"* Dataset path: {data_path}\n"
        return info