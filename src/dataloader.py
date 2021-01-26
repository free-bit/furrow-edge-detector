#!/usr/bin/env python

import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision

from utils.helpers import take_items

ID_FILE = "{folder_id}_ids.txt"
TIME_FILE = "{folder_id}_time.json"

DEPTH_FILE = "{frame_id}_depth.npy"
EDGE_FILE = "{frame_id}_edge_pts.npy"
RGB_FILE = "{frame_id}_rgb.png"
DRGB_FILE = "{frame_id}_depth.png"

class FurrowDataset(Dataset):

    def __init__(self, datapath, **kwargs):
        self.datapath = datapath
        self.validate_datapath()
        
        self.load_darr = kwargs.get("load_darr", True)
        self.load_edge = kwargs.get("load_edge", True)
        self.load_rgb = kwargs.get("load_rgb", False)
        self.load_drgb = kwargs.get("load_drgb", False)
        self.load_time = kwargs.get("load_time", False)
        
        self.frame_ids = []
        self.timestamps = {}
        self.size = 0
        
        start = kwargs.get("start", 0)
        end = kwargs.get("end", np.inf)
        max_frames = kwargs.get("max_frames", np.inf)
        self.read_frame_metadata()
        self.take_frames(start, end, max_frames)

    def validate_datapath(self):
        files = os.listdir(self.datapath)
        for file in files:
            file_path = os.path.join(self.datapath, file)
            assert os.path.isfile(file_path), f"{file_path} is not a file!"
            assert file.endswith(("ids.txt", "depth.npy", "pts.npy", "rgb.png", "depth.png", "vis.png", "time.json")),\
                f"{file} has an unknown extension!"

    def read_frame_metadata(self):
        """Read file ids and timestamps under a single-folder dataset containing frames."""
        folder_id = os.path.basename(self.datapath)

        file = ID_FILE.format(folder_id=folder_id)
        path = os.path.join(self.datapath, file)
        self.frame_ids = np.sort(np.loadtxt(path).astype(np.int64))

        if self.load_time:
            file = TIME_FILE.format(folder_id=folder_id)
            path = os.path.join(self.datapath, file)
            with open(path) as f:
                self.timestamps = json.load(f)

    def take_frames(self, start=0, end=np.inf, max_frames=np.inf):
        """Shrink the number of frames to read according to given range and count"""
        self.frame_ids = take_items(self.frame_ids, start, end, max_frames)

    def __len__(self):
        # Size <-> Number of frames
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        
        item = {
            'frame_id': frame_id,
            'depth_arr': None,
            'edge_pixels': None,
            'rgb_img': None,
            'depth_img': None,
            'time': None
        }

        if self.load_darr:
            darr_file = DEPTH_FILE.format(frame_id=frame_id)
            darr_path = os.path.join(self.datapath, darr_file)
            depth_arr = np.load(darr_path)
            item['depth_arr'] = depth_arr
            
        if self.load_edge:
            edge_file = EDGE_FILE.format(frame_id=frame_id)
            edge_path = os.path.join(self.datapath, edge_file)
            edge_pixels = np.load(edge_path)
            item['edge_pixels'] = edge_pixels
        
        if self.load_rgb:
            rgb_file = RGB_FILE.format(frame_id=frame_id)
            rgb_path = os.path.join(self.datapath, rgb_file)
            rgb_img = Image.open(rgb_path)
            item['rgb_img'] = rgb_img
        
        if self.load_drgb:
            drgb_file = DRGB_FILE.format(frame_id=frame_id)
            drgb_path = os.path.join(self.datapath, drgb_file)
            depth_img = Image.open(drgb_path)
            item['depth_img'] = depth_img

        if self.load_time:
            item['time'] = self.timestamps[str(frame_id)]

        return item

    def __str__(self):
        info = "Status of dataset:\n"+\
               f"* Dataset path: {self.datapath}\n"+\
               f"* Number of frames: {len(self.frame_ids)}\n"+\
               f"* Load depth arrays: {self.load_darr}\n"+\
               f"* Load edge coordinates: {self.load_edge}\n"+\
               f"* Load RGBs: {self.load_rgb}\n"+\
               f"* Load depth RGB files: {self.load_drgb}\n"+\
               f"* Load timestamps: {self.load_time}\n"
        return info

def main():
    datapath = "dataset/20201112_140127"
    args = {
        "load_darr": True,
        "load_edge": False,
        "load_rgb": False,
        "load_drgb": False,
        "load_time": False,
        "start": 0,
        "end": 100,
        "max_frames": np.inf,
    }
    
    dataset = FurrowDataset(datapath, **args)
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
    if args["load_darr"]:
        depth_arr = np.array(item['depth_arr'])
        print(f"Depth array shape: {depth_arr.shape}")

    if args["load_edge"]:
        edge_pixels = item['edge_pixels']
        mask = coord_to_mask(shape, edge_pixels)
        show_image(mask, cmap="gray")

    if args["load_rgb"]:
        rgb_img = np.array(item['rgb_img'])
        show_image(rgb_img)

    if args["load_drgb"]:
        depth_img = np.array(item['depth_img'])
        show_image(depth_img)

    if args["load_time"]:
        time = np.array(item['time'])
        print(f"Timestamp: {time}")

if __name__ == "__main__":
    main()