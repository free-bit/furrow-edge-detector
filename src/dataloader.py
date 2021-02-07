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

ID_FILE = "{folder_id}_ids.txt"
TIME_FILE = "{folder_id}_time.json"

# TODO: For augmented images, add new name templates, e.g. "{frame_id}_shift_edge_pts.npy"
DEPTH_FILE = "{frame_id}_depth.npy"
EDGE_FILE = "{frame_id}_edge_pts.npy"
RGB_FILE = "{frame_id}_rgb.png"
DRGB_FILE = "{frame_id}_depth.png"

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
        t_ids = self.data_args["input_trans"]
        for t_id in t_ids:
            t_list.append(T_MAP[t_id])
        self.input_trans = T.Compose(t_list)
        
        # Output transforms
        t_list = []
        t_ids = self.data_args["output_trans"]
        for t_id in t_ids:
            t_list.append(T_MAP[t_id])
        self.output_trans = T.Compose(t_list)
        
        self.frame_ids = []
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
            assert file.endswith(("ids.txt", "depth.npy", "pts.npy", "rgb.png", "depth.png", "vis.png", "time.json")),\
                f"{file} has an unknown extension!"

    def read_frame_metadata(self):
        """Read file ids and timestamps under a single-folder dataset containing frames."""
        data_path = self.data_args['data_path']
        load_time = self.data_args.get("load_time", False)
        folder_id = os.path.basename(data_path)

        file = ID_FILE.format(folder_id=folder_id)
        path = os.path.join(data_path, file)
        self.frame_ids = np.sort(np.loadtxt(path).astype(np.int64))

        if load_time:
            file = TIME_FILE.format(folder_id=folder_id)
            path = os.path.join(data_path, file)
            with open(path) as f:
                self.timestamps = json.load(f)

    def take_frames(self, start=0, end=np.inf, max_frames=np.inf, step=1):
        """Shrink the number of frames to read according to given range and count"""
        self.frame_ids = take_items(self.frame_ids, start, end, max_frames, step)

    def __len__(self):
        # Size <-> Number of frames
        return len(self.frame_ids)

    def __getitem__(self, idx):
        data_path = self.data_args['data_path']
        shape = self.data_args.get("shape", (480, 640))
        load_darr = self.data_args.get("load_darr", True)
        load_edge = self.data_args.get("load_edge", True)
        load_rgb = self.data_args.get("load_rgb", False)
        load_drgb = self.data_args.get("load_drgb", False)
        load_time = self.data_args.get("load_time", False)
        
        frame_id = self.frame_ids[idx]
        
        item = {
            'frame_id': frame_id,
            # 'depth_arr'
            # 'edge_mask'
            # 'rgb_img'
            # 'depth_img'
            # 'time'
        }

        if load_darr:
            darr_file = DEPTH_FILE.format(frame_id=frame_id)
            darr_path = os.path.join(data_path, darr_file)
            depth_arr = np.load(darr_path) # np.float64
            depth_arr = np.rint(255 * (depth_arr / depth_arr.max()))         # Expand range to [0, 255]
            item['depth_arr'] = self.input_trans(depth_arr.astype(np.uint8)) # np.float64 -> np.uin8
            
        if load_edge:
            edge_file = EDGE_FILE.format(frame_id=frame_id)
            edge_path = os.path.join(data_path, edge_file)
            edge_pixels = np.load(edge_path)
            edge_mask = coord_to_mask(shape, edge_pixels) # PIL.Image (np.uin8)
            item['edge_mask'] = self.output_trans(edge_mask)
        
        if load_rgb:
            rgb_file = RGB_FILE.format(frame_id=frame_id)
            rgb_path = os.path.join(data_path, rgb_file)
            rgb_img = Image.open(rgb_path) # PIL.Image (np.uin8)
            item['rgb_img'] = self.input_trans(rgb_img)
        
        if load_drgb:
            drgb_file = DRGB_FILE.format(frame_id=frame_id)
            drgb_path = os.path.join(data_path, drgb_file)
            depth_img = Image.open(drgb_path) # PIL.Image (np.uin8)
            item['depth_img'] = self.input_trans(depth_img)

        if load_time:
            item['time'] = self.timestamps[str(frame_id)]

        return item

    def __str__(self):
        data_path = self.data_args['data_path']
        load_darr = self.data_args.get("load_darr", True)
        load_edge = self.data_args.get("load_edge", True)
        load_rgb = self.data_args.get("load_rgb", False)
        load_drgb = self.data_args.get("load_drgb", False)
        load_time = self.data_args.get("load_time", False)
        
        info = "Status of dataset:\n"+\
               f"* Dataset path: {data_path}\n"+\
               f"* Number of frames: Fetched/Total: {len(self.frame_ids)}/{self.frame_count}\n"+\
               f"* Load depth arrays: {load_darr}\n"+\
               f"* Load edge coordinates: {load_edge}\n"+\
               f"* Load RGBs: {load_rgb}\n"+\
               f"* Load depth RGB files: {load_drgb}\n"+\
               f"* Load timestamps: {load_time}\n"
        return info

    @staticmethod
    def split_item(item, input_format="darr"):
        samples = {
            'frame_ids': item['frame_id']
        }

        depth_arr = item.get("depth_arr", None)
        rgb_img = item.get("rgb_img", None)
        depth_img = item.get("depth_img", None)
        
        # Depth as array only (C:1) -> (C:3)
        if input_format == "darr":
            samples['input'] = depth_arr.expand(-1, 3, -1, -1) # TODO: This might be made optional

        # RGB image only (C:3)
        elif input_format == "rgb":
            samples['input'] = rgb_img
        
        # Depth as image only  (C:3)
        elif input_format == "drgb":
            samples['input'] = depth_img
        
        # RGB + Depth as array (C:3+1)
        elif input_format == "rgb-darr":
            samples['input'] = torch.cat((rgb_img, depth_arr), dim=1)
        
        # RGB + Depth as image (C:3+3)
        elif input_format == "rgb-drgb":
            samples['input'] = torch.cat((rgb_img, depth_img), dim=1)

        else:
            raise NotImplementedError

        edge_mask = item.get("edge_mask", None)
        samples['gt'] = edge_mask

        return samples

def main():
    data_args = {
        "data_path": "dataset/20201112_140127",
        "load_darr": True,
        "load_edge": False,
        "load_rgb": False,
        "load_drgb": False,
        "load_time": False,
        "start": 0,
        "end": 100,
        "max_frames": np.inf,
    }
    
    dataset = FurrowDataset(data_args)
    print(dataset)
    print(dataset.frame_ids)
    size = len(dataset)

    from utils.helpers import show_image, show_image_pairs, coord_to_mask
    rand_idx = np.random.randint(0, size)
    item = dataset.__getitem__(rand_idx)
    print(item)
    frame_id = item["frame_id"]

    print(f"Random Index: {rand_idx} <-> Frame ID: {frame_id}")

    shape = (480, 640)
    if data_args["load_darr"]:
        depth_arr = np.array(item['depth_arr'])
        print(f"Depth array shape: {depth_arr.shape}")

    if data_args["load_edge"]:
        edge_mask = item['edge_mask']
        show_image(edge_mask, cmap="gray")

    if data_args["load_rgb"]:
        rgb_img = np.array(item['rgb_img'])
        show_image(rgb_img)

    if data_args["load_drgb"]:
        depth_img = np.array(item['depth_img'])
        show_image(depth_img)

    if data_args["load_time"]:
        time = np.array(item['time'])
        print(f"Timestamp: {time}")

if __name__ == "__main__":
    main()