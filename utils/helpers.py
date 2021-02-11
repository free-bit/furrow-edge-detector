## Helper Functions
# Define general purpose methods to here.

import json
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pprint import pprint
import torch
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# TODO: For augmented images, add new name templates, e.g. "{frame_id}_shift_edge_pts.npy"
DEPTH_FILE = "{frame_id}_depth.npy"
EDGE_FILE = "{frame_id}_edge_pts.npy"
RGB_FILE = "{frame_id}_rgb.png"
DRGB_FILE = "{frame_id}_depth.png"

def load_darr(data_path, frame_id):
    darr_file = DEPTH_FILE.format(frame_id=frame_id)
    darr_path = os.path.join(data_path, darr_file)
    depth_arr = np.load(darr_path) # np.float64
    depth_arr = np.rint(255 * (depth_arr / depth_arr.max())) # Expand range to [0, 255]
    return depth_arr.astype(np.uint8) # np.float64 -> np.uin8

def load_rgb(data_path, frame_id):
    rgb_file = RGB_FILE.format(frame_id=frame_id)
    rgb_path = os.path.join(data_path, rgb_file)
    rgb_img = Image.open(rgb_path) # np.uin8
    return np.array(rgb_img)

def load_drgb(data_path, frame_id):
    drgb_file = DRGB_FILE.format(frame_id=frame_id)
    drgb_path = os.path.join(data_path, drgb_file)
    depth_img = Image.open(drgb_path) # np.uin8
    return np.array(depth_img)

def load_edge_coords(data_path, frame_id):
    edge_file = EDGE_FILE.format(frame_id=frame_id)
    edge_path = os.path.join(data_path, edge_file)
    edge_pixels = np.load(edge_path)
    return edge_pixels

def load_edge_mask(data_path, frame_id, shape=(480, 640), edge_width=3):
    edge_pixels = load_edge_coords(data_path, frame_id)
    edge_mask = coord_to_mask(shape, edge_pixels, thickness=edge_width) # np.uin8
    return np.array(edge_mask)

def take_items(items, start=0, end=np.inf, n=np.inf, step=1):
    """Take n items from the specified slice (start,end) and return list of numpy.array or torch.tensor"""
    size = len(items)

    if start < 0:
        start = 0

    if end >= size:
        end = size-1
    
    if start > end or n <= 0:
        return []

    if step < 1:
        step = 1

    items = items[start:end+1:step]
    size = len(items)
    
    if n >= size:
        return list(items)
    
    # Divide items into n splits take one item from each part
    splits = np.array_split(items, n)
    items = [split[0] for split in splits[:-1]] + [splits[-1][-1]]
    return items

def save_config(filename, func_stack, config_stack):
    if not filename.endswith(".json"):
        filename += ".json"
        
    exp = {
        "func_stack": func_stack,
        "config_stack": config_stack,
    }
    
    with open(filename, "w") as file:
        json.dump(exp, file, indent=4)
        
    print(filename, "is created.")
    
def load_config(json_content, func_stack, config_stack):
    print("Previous functions:")
    pprint(func_stack)
    print("Previous configurations:")
    pprint(config_stack)
    
    exp = json.loads(json_content)
    func_stack[:] = exp["func_stack"]
    config_stack[:] = exp["config_stack"]
    
    print("Current functions:")
    pprint(func_stack)
    print("Current configurations:")
    pprint(config_stack)

def update_log(log, stack):
    with log:
        log.clear_output()
        for i, func in enumerate(stack):
            print("{}: {}".format(i, func))

def push(stack, log, e):
    stack.append(e.description)
    update_log(log, stack)

def pop(stack, log):
    if len(stack) > 0:
        stack.pop()
    update_log(log, stack)

def estimate_y(depth, m=0.0027, b=0.06):
    return (1 / depth - b) / m

def estimate_depth(y, m=0.0027, b=0.06):
    return 1 / (m * y + b)

def estimate_dy(depth0, ddepth, m=0.0027, b=0.06):
    y0 = estimate_y(depth0, m, b)
    depth1 = depth0 + ddepth
    y1 = estimate_y(depth1, m, b)
    return abs(y1-y0)

def estimate_ddepth(depth0, dy, m=0.0027, b=0.06):
    y0 = estimate_y(depth0, m, b)
    y1 = y0 - dy
    depth1 = estimate_depth(y1, m, b)
    return abs(depth1-depth0)

def parabola(y, a, b, c):
    return (a * y**2) + (b * y) + c

# def parabola(y, *coeff):
#     pow = np.arange(len(coeff))[::-1]
#     result = (coeff * y**pow).sum()
#     return result

def apply_geometric_formula(size, p):
    """Given a geometric shape parametrized by 'p', returns pixel coordinates."""
    y_min, y_max = 0, size[0]
    y = np.arange(y_min, y_max, 1)
    x = None

    if p['type'] == "slope":
        m, b = p['m'], p['b']
        x = m * y + b

    elif p['type'] == "symmetric":
        v0, v1 = p['v']
        x0, y0 = p['p']
        x = (y - y0) * v0 / v1 + x0

    elif p['type'] == "hough":
        rho, theta = p['rho'], p['theta']
        if theta == np.pi/2:
            x_min, x_max = 0, size[1]
            y = np.ones(size[1], dtype=np.int64) * rho
            x = np.arange(x_min, x_max, 1)
        else:
            x = (rho - y * np.sin(theta)) / np.cos(theta)

    elif p['type'] == "parabola":
        a, b, c = p['a'], p['b'], p['c']
        x = parabola(y, a, b, c)

    else:
        raise NotImplementedError
    
    x = np.rint(x).astype(np.int64)
    pixel_coords = np.stack((y,x), axis=1)
    return pixel_coords

def compute_visible_pixels(size, p):
    """Given a geometric shape parametrized by 'p', returns coordinates for visible line pixels (Nx2)."""
    # Compute pixels belonging to the line
    pixel_coords = apply_geometric_formula(size, p)
    
    # Take coordinates in the 1st quadrant
    pixel_coords = pixel_coords[np.min(pixel_coords, axis=1)>=0]
    
    # Take coordinates in the visible region
    visible = (pixel_coords[:,0] < size[0]) & (pixel_coords[:,1] < size[1])
    pixel_coords = pixel_coords[visible]
    
    return pixel_coords

def coord_to_mask(shape, yx, thickness=1):
    """Convert pixel coordinates of a shape into a binary mask"""
    xy = yx[:, [1,0]]
    mask = np.zeros(shape, dtype=np.uint8)
    mask = cv2.polylines(mask, [xy], False, 255, thickness, cv2.LINE_8)
    mask = (mask.astype(np.bool) * 255).astype(np.uint8)
    return mask

def set_roi(image, num_corners=5):
    """Choose corners for ROI interactively, returns ROI corners in pixel coordinates"""
    def info(msg):
        plt.title(msg, fontsize=15)
        plt.draw()
        
    roi_corners = []
    handler = plt.imshow(image)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()

    info('Click to start...')    
    plt.waitforbuttonpress()
    
    info('Select {} corners with mouse.'.format(num_corners))
    roi_corners = np.array(plt.ginput(num_corners, timeout=-1))

    C = image.shape[2] if len(image.shape) == 3 else 1
    black = (0,) * C
    roi_corners = np.rint(roi_corners).astype(np.int32)
    image = cv2.polylines(image.copy(), [roi_corners], isClosed=True, color=black)
    handler.set_data(image)
        
    info('Click to quit...')
    plt.waitforbuttonpress()
    
    return roi_corners
    
def show_image(image, h=10, w=10, cmap=None):
    plt.figure(figsize=(w,h))
    plt.imshow(image, cmap=cmap)
    plt.show()
    
def show_shapes(image, shapes2pixels, shapeIdx='all', cmap=None):
    """
    shapes2pixels is a list containing a 2D array for each shape.
    The 2D array for a shape is has the dimension Px2 where: 
    * P rows denote the pixels of the shape
    * 2 columns are for the coordinates of a pixel, in the form: (row, col). 
    """
    
    # Swap columns for coordinates: (row/y, col/x) -> (x, y)
    shapes2pixels_cv = shapes2pixels.copy()
    for i in range(len(shapes2pixels_cv)):
        shapes2pixels_cv[i] = shapes2pixels_cv[i][:, [1,0]]
    
    # To draw all the contours in the image, set contourIdx to -1
    if shapeIdx == 'all':
        shapeIdx = -1
        
    C = image.shape[2] if len(image.shape) == 3 else 1
    white = (255,) * C

    image = cv2.drawContours(image, shapes2pixels_cv, contourIdx=shapeIdx, color=white, thickness=1)
    show_image(image, cmap=cmap)

def show_corners(image, corners, h=10, w=10, cmap="gray"):
    plt.figure(figsize=(w,h))
    plt.imshow(image, cmap="gray")
    plt.scatter(corners[:, 1], corners[:, 0], color="red", marker="x")
    plt.show()

def show_overlay(image, mask):
    """Overlay a binary mask on top of RGB image"""
    overlaid = image.copy()
    color = (0, 0, 0) if len(image.shape) == 3 else 255
    overlaid[mask == 255] = color
    show_image(overlaid)

def show_image_pairs(left, right, h=15, w=15):
    f, ax = plt.subplots(1,2)
    f.set_figheight(h)
    f.set_figwidth(w)
    ax[0].imshow(left)
    ax[1].imshow(right)
    plt.show()

def show_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    
    plt.title('Hierarchical Clustering Dendrogram')
    
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

def cluster_horizontally(corners, n_clusters=None, distance_threshold=100, linkage="average", verbose=0):
    if len(corners.shape) == 3:
        corners = corners.reshape(-1, 2)
    _, x = np.hsplit(corners, 2)

    model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, linkage=linkage)
    label_per_item = model.fit_predict(x)
    
    n_clusters = model.n_clusters_
    print("[Info]: Number of estimated clusters : %d" % n_clusters)
    
    labels, items_per_label = np.unique(label_per_item, return_counts=True)
    inlier_label = np.argmax(items_per_label)
    outlier_labels = np.delete(labels, inlier_label)
    print("[Info]: Inlier count: {}".format(items_per_label[inlier_label]))
    print("[Info]: Outlier count: {}".format(items_per_label[outlier_labels].sum()))
    
    inlier_idxs = (label_per_item == inlier_label)
    inlier_corners = corners[inlier_idxs]
    
    if verbose >= 2:
        outlier_idxs = ~inlier_idxs
        outliers = corners[outlier_idxs]
        plt.title('Inlier: [{}] vs Outliers: {}'.format(inlier_label, outlier_labels))
        inlier_pts = plt.scatter(inlier_corners[:,1], inlier_corners[:,0])
        outlier_pts = plt.scatter(outliers[:,1], outliers[:,0])
        plt.legend((inlier_pts, outlier_pts), (f"inlier ([{inlier_label}])", "outlier ({})".format(outlier_labels)), loc=4)
        plt.show()
    
    if verbose >= 4:
        print("[Info]: Items per cluster:", items_per_label)
        plots = []
        for label in labels:
            item_idxs = (label_per_item == label)
            items = corners[item_idxs]
            plot = plt.scatter(items[:,1], items[:,0])
            plots.append(plot)
        plt.legend(plots, labels, loc=4)
        plt.show()

    if verbose >= 3:
        show_dendrogram(model, truncate_mode='level', p=3)
    
    return inlier_corners.reshape(-1, 1, 2)

def create_template(size, position=4):
    """
    [1 | 2]
    -------
    [3 | 4]
    """
    mask = np.zeros((size, size), dtype=np.float32)
    if position == 1:
        mask[:size//2,:size//2] = 1
    elif position == 2:
        mask[:size//2,size//2:] = 1
    elif position == 3:
        mask[size//2:,:size//2] = 1
    elif position == 4:
        mask[size//2:,size//2:] = 1
    else:
        raise NotImplementedError
    return mask

def generate_lane_pixels(left, right, pixel_offset=0, num_lane=15):
    mid = np.rint((left + right) / 2).astype(np.int32)
    mid = mid[pixel_offset:]
    segments = np.array_split(mid, num_lane)
    return np.concatenate(segments[::2], axis=0)

def shift_pixels(depth_arr, pixel_coords, intrinsics, shift3D=[-0.25, 0, 0]):
    """
    Given a translation vector in 3D in meters, compute new pixel coordinates.
    Instead of doing a 3D translation followed by a projection (a),
    translation vector is projected and projected vector is applied on 2D (b).
    (a) 2D -> 3D translate -> 2D
      1) X1 = (x1 - ppx) / fx * depth
      2) fx * (X1 - 0.25) / depth + ppx
      * 1 & 2 yields round(x1 - fx * 0.25 / depth)

    (b) 3D -> 2D translate (projected)
      1) dx = fx * 0.25 / depth
      2) round(x1 - dx)
    """
    depths = depth_arr[pixel_coords[:,0], pixel_coords[:,1]]
    dx, dy, dz = shift3D
    fx, fy = intrinsics.fx, intrinsics.fy
    depths += dz
    dx = fx * dx / depths
    dy = fy * dy / depths
    shifted = np.rint(pixel_coords + np.c_[dy, dx]).astype(np.int32)
    return shifted

def project(cam_coords, intrinsics):
    z = cam_coords[:, 2]
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy
    x = np.rint(cam_coords[:,0] * fx / z + ppx).astype(np.int32)
    y = np.rint(cam_coords[:,1] * fy / z + ppy).astype(np.int32)
    return np.c_[y, x]

def backproject(depth_arr, pixel_coords, intrinsics):
    z = depth_arr[pixel_coords[:,0], pixel_coords[:,1]]
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy
    x = (pixel_coords[:,1] - ppx) / fx * z
    y = (pixel_coords[:,0] - ppy) / fy * z
    return np.c_[x, y, z]

def print_full(arr, outfile="temp.csv"):
    header = list(map(str, np.arange(1, arr.shape[1] + 1, 1)))
    print(list(header))
    pd.DataFrame(arr).to_csv(outfile, header=header, index=False, index_label=True)

def colorize(depth_arr):
    # Initialize disparity with zeros, take 1/depth_arr where depth_arr!=0
    disparity = np.divide(1, depth_arr, out=np.zeros_like(depth_arr), where=depth_arr!=0)
    cm = plt.get_cmap('jet')
    depth_image = cm(disparity)
    plt.imshow(depth_image)
    plt.show()