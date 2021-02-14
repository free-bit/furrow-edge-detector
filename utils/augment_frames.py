#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from PIL import Image
from scipy.ndimage import shift, rotate

from utils.helpers import coord_to_mask, prepare_overlay

AUG_LIST = ['m', 't', 'r', 't-r', 'r-t', 'm-t', 'm-r', 'm-t-r', 'm-r-t']

def sample_from_range(low, high):
    values = list(range(low, high+1))
    values.remove(0)
    return np.random.choice(values)

def mirror_item(item, shape=(480, 640)):
    """Return a mirrored copy of the item."""
    print("Flip about y-axis")
    modified = {'frame_id': item['frame_id']}
    
    if "depth_arr" in item:
        modified["depth_arr"] = np.fliplr(item["depth_arr"])
    
    if "rgb_img" in item:
        as_array = np.array((item["rgb_img"]))
        modified["rgb_img"] = Image.fromarray(np.fliplr(as_array))
    
    if "depth_img" in item:
        as_array = np.array((item["depth_img"]))
        modified["depth_img"] = Image.fromarray(np.fliplr(as_array))
    
    if "edge_pixels" in item:
        modified["edge_pixels"] = np.abs(item["edge_pixels"] - [0, shape[1]-1])
    
    return modified

def translate_item(item, tx, ty=0):
    """Return a translated copy of the item."""
    print(f"Translation with: ({ty}, {tx}) pixels")
    modified = {'frame_id': item['frame_id']}

    if "edge_pixels" in item:
        modified["edge_pixels"] = item["edge_pixels"] + [ty, tx]

    if "depth_arr" in item:
        modified["depth_arr"] = shift(item["depth_arr"], [ty,tx])

    if "rgb_img" in item:
        as_array = np.array(item["rgb_img"])
        modified["rgb_img"] = Image.fromarray(shift(as_array, [ty,tx,0]))

    if "depth_img" in item:
        as_array = np.array(item["depth_img"])
        modified["depth_img"] = Image.fromarray(shift(as_array, [ty,tx,0]))

    return modified

def rotate_item(item, degrees, shape=(480, 640)):
    """Return a rotated copy of the item."""
    print(f"Rotation with: {degrees} degrees (ccw)")
    modified = {'frame_id': item['frame_id']}
    
    if "depth_arr" in item:
        modified["depth_arr"] = rotate(item["depth_arr"], degrees, reshape=False, order=0)
    
    if "rgb_img" in item:
        as_array = item["rgb_img"]
        modified["rgb_img"] = Image.fromarray(rotate(as_array, degrees, reshape=False, order=0))
    
    if "depth_img" in item:
        as_array = item["depth_img"]
        modified["depth_img"] = Image.fromarray(rotate(as_array, degrees, reshape=False, order=0))
    
    if "edge_pixels" in item:
        edge_pixels = item["edge_pixels"]
        edge_mask = coord_to_mask(shape, edge_pixels, thickness=1)
        rot_edge_mask = rotate(edge_mask, degrees, reshape=False, order=0)
        edge_pixels = np.stack(np.where(rot_edge_mask == 255), axis=1)
        modified["edge_pixels"] = edge_pixels
    
    return modified

def apply_random_augment(item, **kwargs):
    tx_low = kwargs.get('tx_low', -230)
    tx_high = kwargs.get('tx_high', 170)
    deg_low = kwargs.get('deg_low', -10)
    deg_high = kwargs.get('deg_high', 10)

    tag = "_"
    modified = item

    augs = np.random.choice(AUG_LIST).split('-')
    for aug in augs:
        if aug == 'm':
            modified = mirror_item(modified)
            tag += "m_"
            tx_low, tx_high = -tx_high, -tx_low
        
        elif aug == 't':
            tx = sample_from_range(tx_low, tx_high)
            modified = translate_item(modified, tx, ty=0)
            tag += f"t{tx}_"

        elif aug == 'r':
            degrees = sample_from_range(deg_low, deg_high)
            modified = rotate_item(modified, degrees)
            tag += f"r{degrees}_"
        
        else:
            raise NotImplementedError
    
    modified['tag'] = tag
    return modified

def store_item(item, path):
    tag = item['tag']
    frame_id = item['frame_id']
    
    if "edge_pixels" in item:
        filename = str(frame_id) + tag + "edge_pts.npy"
        np.save(os.path.join(path, filename), item["edge_pixels"])

    if "depth_arr" in item:
        filename = str(frame_id) + tag + "depth.npy"
        np.save(os.path.join(path, filename), item["depth_arr"])

    if "rgb_img" in item:
        filename = str(frame_id) + tag + "rgb.png"
        item["rgb_img"].save(os.path.join(path, filename))

    if "depth_img" in item:
        filename = str(frame_id) + tag + "depth.png"
        item["depth_img"].save(os.path.join(path, filename))

    # if "rgb_img" in item and "edge_pixels" in item:
    #     rgb_img = np.array(item['rgb_img'])
    #     edge_pixels = item['edge_pixels']
    #     edge_mask = coord_to_mask(rgb_img.shape, edge_pixels, thickness=1)
    #     overlaid = prepare_overlay(rgb_img, edge_mask)
    #     overlaid = Image.fromarray(overlaid)
        
    #     filename = str(frame_id) + tag + "overlay.png"
    #     overlaid.save(os.path.join(path, filename))

# Naming convention: 
# frameid_tag_type, e.g. 3900_m_t-5_r10_edge_pts.npy
# tag: transform sequence from left to right
# type: file type and extension
# Other possible strategy for shift:
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