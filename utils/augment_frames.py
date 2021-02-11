#!/usr/bin/env python
# coding: utf-8

import argparse

from scipy.ndimage import shift, rotate
import numpy as np
from PIL import Image, ImageOps

from utils.helpers import coord_to_mask

# Handles command line arguments
def arg_handler():
    parser = argparse.ArgumentParser(description='Augment provided dataset', 
                                     add_help=True)
    # Optional flags
    optional = parser._optionals
    optional.title = 'Optional arguments'
    optional._option_string_actions["-h"].help = "Show this help message and exit"

    optional.add_argument("--nshift", help="", type=int)

    optional.add_argument("--mirror", help="", default=False)

    optional.add_argument("--nrotate", help="", type=int)
    
    # optional.add_argument("--out",
    #                    type=str,
    #                    help="Specify output folder name (default: './extracted')", 
    #                    metavar="FOLDER",
    #                    default="./augmented")


    required = parser.add_argument_group(title='Required arguments')

    required.add_argument("--in",
                       type=str,
                       help="Specify input folder name", 
                       metavar="FOLDER",
                       required=True)

    args = parser.parse_args()

    return args

def mirror_item(item):
    modified = {}
    
    if "depth_arr" in item:
        modified["depth_arr"] = np.fliplr(item["depth_arr"])
    
    if "rgb_img" in item:
        modified["rgb_img"] = ImageOps.mirror(item["rgb_img"])
    
    if "depth_img" in item:
        modified["depth_img"] = ImageOps.mirror(item["depth_img"])
    
    if "edge_pixels" in item:
        modified["edge_pixels"] = np.abs(item["edge_pixels"] - [0, 639])
    
    return modified

def translate_item(item, tx, ty=0):
    print(f"Translation with: ({ty}, {tx}) pixels")
    modified = {}
    shape = (480, 640)
    # max_right = np.inf
    # max_left = -np.inf

    if "edge_pixels" in item:
        # min_x = item["edge_pixels"].min(axis=0)[1]
        # max_x = item["edge_pixels"].max(axis=0)[1]
        # mid_x = (max_x + min_x) / 2
        # margin = (max_x - mid_x) / 2
        # max_right = 640 - margin - mid_x
        # max_left = margin - mid_x
        # print("max_x:", max_x)
        # print("mid_x:", mid_x)
        # print("min_x:", min_x)
        # print("margin:", margin)
        # print("max_right:", max_right)
        # print("max_left:", max_left)
        # tx = np.random.randint(max_left, max_right+1)
        # print("tx:", tx)

        modified["edge_pixels"] = item["edge_pixels"] + [ty, tx]

    if "depth_arr" in item:
        modified["depth_arr"] = shift(item["depth_arr"], [ty,tx])

    if "rgb_img" in item:
        image = item["rgb_img"]
        modified["rgb_img"] = image.transform(image.size, Image.AFFINE, data=[1, 0, -tx, 0, 1, -ty, 0, 0, 1])

    if "depth_img" in item:
        image = item["depth_img"]
        modified["depth_img"] = image.transform(image.size, Image.AFFINE, data=[1, 0, -tx, 0, 1, -ty, 0, 0, 1])
    
    shift_mask = np.ones(shape, dtype=np.uint8) * 255
    modified['shift_mask'] = shift(shift_mask, [ty,tx])

    return modified

def rotate_item(item, degrees):
    print(f"Rotation with: {degrees} degrees")
    modified = {}
    shape = (480, 640)
    
    if "depth_arr" in item:
        modified["depth_arr"] = rotate(item["depth_arr"], degrees, reshape=False)
    
    if "rgb_img" in item:
        modified["rgb_img"] = item["rgb_img"].rotate(degrees)
    
    if "depth_img" in item:
        modified["depth_img"] = item["depth_img"].rotate(degrees)
    
    if "edge_pixels" in item:
        edge_pixels = item["edge_pixels"]
        edge_mask = coord_to_mask(shape, edge_pixels, thickness=1)
        rot_edge_mask = rotate(edge_mask, degrees, reshape=False)
        edge_pixels = np.stack(np.where(rot_edge_mask == 255), axis=1)
        modified["edge_pixels"] = edge_pixels

    rot_mask = np.ones(shape, dtype=np.uint8) * 255
    modified['rot_mask'] = rotate(rot_mask, degrees, reshape=False)
    
    return modified

def sample_from_range(val_range):
    return np.random.randint(val_range[0], val_range[1]+1)