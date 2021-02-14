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

T_MAP = {
    "affine": F.affine,
    "center_crop": T.Lambda(lambda x:  F.center_crop(x, output_size=[400,400])),
    "crop_right": T.Lambda(lambda x: F.crop(x, top=80, left=240, height=400, width=400)),
    "crop_left": T.Lambda(lambda x: F.crop(x, top=80, left=0, height=400, width=400)),
    "gaussian_blur": F.gaussian_blur,
    "normalize_imagenet": T.Lambda(lambda x: F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    "normalize_furrowset": T.Lambda(lambda x: F.normalize(x, mean=0.5, std=0.5)), # Pixel values in range: [-1,1] # TODO: Compute mean & variance in the dataset
    "resize": F.resize,
    "rotate": F.rotate,
    "to_tensor": F.to_tensor, # Pixel values in range: [0,1]
}
# TODO: Test new dataset implementation, then extract new augmentations
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
        folder_id = os.path.basename(data_path)
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
            file = str(folder_id) + TIME_EXT
            path = os.path.join(data_path, file)
            with open(path) as f:
                self.timestamps = json.load(f)

    def take_frames(self, start=0, end=np.inf, max_frames=np.inf, step=1):
        """Shrink the number of frames to read according to given range and count"""
        self.frame_ids = take_items(self.frame_ids, start, end, max_frames, step)
        self.augs = take_items(self.augs, start, end, max_frames, step)

    def __len__(self):
        # Size <-> Number of frames
        return len(self.frame_ids)

    def get_frame_files(self, idx, load_darr, load_rgb, load_drgb, load_edge, load_time):
        # General purpose method to load data only
        data_path = self.data_args['data_path']
        
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
            frame_files['edge_pixels'] = np.load(edge_file)  # np.array: np.int64

        # Load depth as array only
        if load_darr:
            darr_file = str(frame_id) + tag + DEPTH_EXT
            darr_file = os.path.join(data_path, darr_file)
            frame_files['depth_arr'] = np.load(darr_file)    # np.array: np.float64

        # Load RGB image only
        if load_rgb:
            rgb_file = str(frame_id) + tag + RGB_EXT
            rgb_file = os.path.join(data_path, rgb_file)
            frame_files['rgb_img'] = Image.open(rgb_file)    # PIL.Image: np.uint8
        
        # Load depth as image only
        if load_drgb:
            drgb_file = str(frame_id) + tag + DRGB_EXT
            drgb_file = os.path.join(data_path, drgb_file)
            frame_files['depth_img'] = Image.open(drgb_file) # PIL.Image: np.uint8

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
        sample = {}
        
        # Target: edge_pixels -> edge_mask -> transform (if loaded)
        if load_edge:
            edge_pixels = frame_files['edge_pixels']
            edge_mask = coord_to_mask(shape, edge_pixels, thickness=edge_width) # np.uint8
            sample['target'] = self.target_trans(edge_mask)

            # TODO: Keep or discard If data is augmented, compute a region of interest where loss is to be calculated.
            # loss_mask = np.ones(shape, dtype=np.bool)
            # augs = frame_files['augs']
            # for aug in augs:
            #     aug_type = aug[0]
                
            #     if aug_type == 't':
            #         tx = int(aug[1:])
            #         loss_mask = shift(loss_mask, [0,tx])
                
            #     elif aug_type == 'r':
            #         degrees = int(aug[1:])
            #         loss_mask = rotate(loss_mask, degrees, reshape=False, order=0)
            
            # sample['loss_mask'] = self.target_trans(loss_mask)
        
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
               f"* Number of frames loaded/total: {len(self.frame_ids)}/{self.frame_count}\n"+\
               f"* Input format: {input_format}\n"+\
               f"* Load depth as array: {load_darr}\n"+\
               f"* Load RGB: {load_rgb}\n"+\
               f"* Load depth as RGB: {load_drgb}\n"+\
               f"* Load edge coordinates (ground truth): {load_edge}\n"+\
               f"* Load timestamps: {load_time}\n"
        return info