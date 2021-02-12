#!/usr/bin/env python

import json
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from utils.helpers import coord_to_mask, take_items

TIME_FILE = "{folder_id}_time.json"

T_MAP = {
    "affine": F.affine,
    "center_crop": F.center_crop,
    "crop_right": T.Lambda(lambda x: F.crop(x, top=80, left=240, height=400, width=400)), 
    "crop_left": T.Lambda(lambda x: F.crop(x, top=80, left=0, height=400, width=400)),
    "gaussian_blur": F.gaussian_blur,
    "normalize_imagenet": T.Lambda(lambda x: F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    "normalize_furrowset": T.Lambda(lambda x: F.normalize(x, mean=0.5, std=0.5)), # Pixel values in range: [-1,1] # TODO: Compute mean & variance in the dataset
    "resize": F.resize,
    "rotate": F.rotate,
    "to_tensor": F.to_tensor, # Pixel values in range: [0,1]
}

class FurrowDataset(Dataset):

    def __init__(self, data_args):
        self.data_args = data_args
        self.validate_data_path()

        # Input transforms
        t_list = []
        t_ids = self.data_args.get("input_trans", [])
        for t_id in t_ids:
            t_list.append(T_MAP[t_id])
        self.input_trans = T.Compose(t_list)
        
        # Output transforms
        t_list = []
        t_ids = self.data_args.get("target_trans", [])
        for t_id in t_ids:
            t_list.append(T_MAP[t_id])
        self.target_trans = T.Compose(t_list)
        
        self.unique_frame_ids = []
        self.timestamps = {}
        self.rgb_files = []
        self.darr_files = []
        self.edge_files = []
        self.drgb_files = []
        self.size = 0
        
        start = data_args.get("start", 0)
        end = data_args.get("end", np.inf)
        max_frames = data_args.get("max_frames", np.inf)
        step = data_args.get("step", 1)
        self.read_frame_metadata()
        self.unique_frame_count = len(self.unique_frame_ids)
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
        folder_id = os.path.basename(data_path)
        files = os.listdir(data_path)
        
        # Build an index set for frames available
        frame_ids = set()
        for file in files:
            metadata = file.split("_")
            frame_id = int(metadata[0])
            frame_ids.add(frame_id)
        
        # Filter out timestamp file from index (if exists)
        try:
            prefix = int(folder_id.split("_")[0])
            frame_ids.remove(prefix)
        except ValueError:
            pass

        self.unique_frame_ids = list(frame_ids)

        if load_time:
            file = TIME_FILE.format(folder_id=folder_id)
            path = os.path.join(data_path, file)
            with open(path) as f:
                self.timestamps = json.load(f)

    def take_frames(self, start=0, end=np.inf, max_frames=np.inf, step=1):
        """Shrink the number of frames to read according to given range and count"""
        data_path = self.data_args['data_path']
        files = os.listdir(data_path)
        self.unique_frame_ids = take_items(self.unique_frame_ids, start, end, max_frames, step)
        
        selection = list(filter(lambda x: int(x.split("_")[0]) in self.unique_frame_ids, files))
        rgb_files = list(filter(lambda x: x[-len('rgb.png'):] == 'rgb.png', selection))
        darr_files = list(filter(lambda x: x[-len('depth.npy'):] == 'depth.npy', selection))
        edge_files = list(filter(lambda x: x[-len('edge_pts.npy'):] == 'edge_pts.npy', selection))
        drgb_files = list(filter(lambda x: x[-len('depth.png'):] == 'depth.png', selection))
        self.rgb_files = sorted(rgb_files, key=lambda x: int(x.split("_")[0]))
        self.darr_files = sorted(darr_files, key=lambda x: int(x.split("_")[0]))
        self.edge_files = sorted(edge_files, key=lambda x: int(x.split("_")[0]))
        self.drgb_files = sorted(drgb_files, key=lambda x: int(x.split("_")[0]))

    def __len__(self):
        # Size <-> Number of frames (+ augmentations)
        return len(self.rgb_files)

    def get_frame_files(self, idx, load_darr, load_rgb, load_drgb, load_edge, load_time):
        # General purpose method to load data only
        data_path = self.data_args['data_path']
        frame_id = int(self.rgb_files[idx].split("_")[0])
        frame_files = {
            "frame_id": frame_id
        }

        # Load edge mask
        if load_edge:
            edge_file = os.path.join(data_path, self.edge_files[idx])
            frame_files['edge_pixels'] = np.load(edge_file)  # np.array: np.int64

        # Load depth as array only
        if load_darr:
            darr_file = os.path.join(data_path, self.darr_files[idx])
            frame_files['depth_arr'] = np.load(darr_file)    # np.array: np.float64

        # Load RGB image only
        if load_rgb:
            rgb_file = os.path.join(data_path, self.rgb_files[idx])
            frame_files['rgb_img'] = Image.open(rgb_file)    # PIL.Image: np.uint8
        
        # Load depth as image only
        # if load_drgb:
        #     drgb_file = os.path.join(data_path, self.drgb_files[idx])
        #     frame_files['depth_img'] = Image.open(drgb_file) # PIL.Image: np.uint8

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
        
        frame_files = self.get_frame_files(idx, load_darr, load_rgb, load_drgb, load_edge, load_time)
        sample = {
            "frame_id": frame_files["frame_id"]
        }
        
        # Target: edge_pixels -> edge_mask -> transform (if loaded)
        if load_edge:
            edge_pixels = frame_files['edge_pixels']
            edge_mask = coord_to_mask(shape, edge_pixels, thickness=edge_width) # np.uint8
            sample['target'] = self.target_trans(edge_mask)
        
        # Input Option-1: depth_arr: (C:1) -> (C:3) -> transform (if loaded)
        if input_format == "darr":
            depth_arr = frame_files['depth_arr']
            depth_arr = np.rint(255 * (depth_arr / depth_arr.max()))         # Expand range to [0, 255]
            depth_arr = depth_arr.astype(np.uint8)                           # np.float64 -> np.uint8
            depth_arr = np.stack([depth_arr, depth_arr, depth_arr], axis=-1) # (C:1) -> (C:3)
            sample['input'] = self.input_trans(depth_arr) # TODO: This might be made optional

        # Input Option-2: rgb_img: (C:3) -> transform (if loaded)
        elif input_format == "rgb":
            sample['input'] = self.input_trans(frame_files['rgb_img'])
        
        # Input Option-3: depth_img: (C:3) -> transform (if loaded)
        elif input_format == "drgb":
            sample['input'] = self.input_trans(frame_files['depth_img'])
        
        # Input Option-4: rgb_img + depth_arr: (C:3) + (C:1) -> (C:4) -> transform (if loaded)
        elif input_format == "rgb-darr":
            rgb_img = self.input_trans(frame_files['rgb_img'])
            depth_arr = self.input_trans(frame_files['depth_arr'])
            sample['input'] = torch.cat((rgb_img, depth_arr), dim=-1)
        
        # Input Option-5: rgb_img + depth_img: (C:3) + (C:3) -> (C:6) -> transform (if loaded)
        elif input_format == "rgb-drgb":
            rgb_img = self.input_trans(frame_files['rgb_img'])
            depth_img = self.input_trans(frame_files['depth_img'])
            sample['input'] = torch.cat((rgb_img, depth_img), dim=-1)

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
        
        info = "Status of dataset:\n"+\
               f"* Dataset path: {data_path}\n"+\
               f"* Number of files loaded: : {len(self)}\n"+\
               f"* Number of frames: Fetched/Total: {len(self.unique_frame_ids)}/{self.unique_frame_count}\n"+\
               f"* Input format: {input_format}\n"+\
               f"* Load depth as array: {load_darr}\n"+\
               f"* Load RGB: {load_rgb}\n"+\
               f"* Load depth as RGB: {load_drgb}\n"+\
               f"* Load edge coordinates (ground truth): {load_edge}\n"+\
               f"* Load timestamps: {load_time}\n"
        return info