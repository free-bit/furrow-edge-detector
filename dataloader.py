#!/usr/bin/env python

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision

from utils import take_items

class FurrowDataset(Dataset):

    def __init__(self, folder, **kwargs):
        self.folder = folder
        
        self.load_darr = kwargs.get("load_darr", True)
        self.load_edge = kwargs.get("load_edge", True)
        self.load_rgb = kwargs.get("load_rgb", False)
        self.load_drgb = kwargs.get("load_drgb", False)
        self.load_time = kwargs.get("load_time", False)
        
        self.darr_files = []
        self.edge_files = []
        self.rgb_files = []
        self.drgb_files = []
        self.time_file = None
        self.size = 0
        
        start = kwargs.get("start", 0)
        end = kwargs.get("end", np.inf)
        max_frames = kwargs.get("max_frames", np.inf)
        self.read_folder(start, end, max_frames)

    def read_folder(self, start=0, end=np.inf, max_frames=np.inf):
        """Return file names under a single folder."""
        files = os.listdir(self.folder)

        file_check = lambda file: os.path.isfile(os.path.join(self.folder, file))
        assert all(map(file_check, files)), f"Error: There are folders under directory: {self.folder}"

        # Filter files wrt their extension
        for file in files:            
            if self.load_darr and file.endswith("depth.npy"):
                self.darr_files.append(file)
            
            elif self.load_edge and file.endswith("edge_pts.npy"):
                self.edge_files.append(file)
            
            elif self.load_rgb and file.endswith("rgb.png"):
                self.rgb_files.append(file)
            
            elif self.load_drgb and file.endswith("depth.png"):
                self.drgb_files.append(file)
            
            elif self.load_time and file.endswith(".json"):
                self.time_file = file

        # Take max_frames (if specified)
        if self.load_darr:
            self.darr_files = sorted(self.darr_files, key=lambda f: int(f.split("_")[0]))
            self.darr_files = take_items(self.darr_files, start, end, max_frames)
        
        if self.load_edge:
            self.edge_files = sorted(self.edge_files, key=lambda f: int(f.split("_")[0]))
            self.edge_files = take_items(self.edge_files, start, end, max_frames)
        
        if self.load_rgb:
            self.rgb_files = sorted(self.rgb_files, key=lambda f: int(f.split("_")[0]))
            self.rgb_files = take_items(self.rgb_files, start, end, max_frames)
        
        if self.load_drgb:
            self.drgb_files = sorted(self.drgb_files, key=lambda f: int(f.split("_")[0]))
            self.drgb_files = take_items(self.drgb_files, start, end, max_frames)

        # Size <-> Number of frames
        self.size = max(len(self.darr_files), len(self.edge_files), len(self.rgb_files), len(self.drgb_files))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        pass

    def __str__(self):
        info = "Status of dataset:\n"+\
               f"* Folder: {self.folder}\n"+\
               f"* Depth array files loaded: {len(self.darr_files)}\n"+\
               f"* Edge coordinate files loaded: {len(self.edge_files)}\n"+\
               f"* RGB files loaded: {len(self.rgb_files)}\n"+\
               f"* Depth RGB files loaded: {len(self.drgb_files)}\n"+\
               f"* Timestamp files loaded: {1 if self.time_file else 0}\n"
        return info

def main():
    folder = "front(20201112_140127)"
    args = {
        "load_darr": False,
        "load_edge": True,
        "load_rgb": False,
        "load_drgb": True,
        "load_time": False,
        "start": 0,
        "end": np.inf,
        "max_frames": 20,
    }
    dataset = FurrowDataset(folder, **args)
    print(dataset)
    print("Frame count:", len(dataset))
    print("Depth array files:", dataset.darr_files[:20])
    print("Edge coordinate files:", dataset.edge_files[:20])
    print("RGB files:", dataset.rgb_files[:20])
    print("Depth RGB files:", dataset.drgb_files[:20])
    print("Timestamp files:", dataset.time_file)

if __name__ == "__main__":
    main()