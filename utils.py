## Helper Functions
# Define general purpose methods to here.

import json

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
from pprint import pprint
from scipy.cluster.hierarchy import dendrogram
from skimage.draw import line
from sklearn.cluster import AgglomerativeClustering

import image_processing

def top_n_idxs(arr, n):
    """Find top N elements in D dimensional array and returns indices (NxD)"""
    flat_arr = np.ravel(arr)
    # np.argsort sorts in ascending order, take last n elements in reverse order
    flat_top_idxs = np.argsort(flat_arr)[-1:-(n+1):-1]
    top_idxs = np.unravel_index(flat_top_idxs, arr.shape)
    return np.array(list(zip(*top_idxs)))

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


def line_endpoints(shape, p):
    """Given a line parametrized by 'p', returns coordinates for endpoints for the visible line segment (x0,y0), (x1,y1)."""
    y_min, y_max = 0, shape[0] - 1
    p1, p2 = None, None

    if p['type'] == "slope":
        m, b = p['m'], p['b']
        p1 = (y_min, (m * y_min + b))
        p2 = (y_max, (m * y_max + b))

    elif p['type'] == "symmetric":
        v0, v1 = p['v']
        x0, y0 = p['p']
        p1 = (y_min, (y_min - y0) * v0 / v1 + x0)
        p2 = (y_max, (y_max - y0) * v0 / v1 + x0)

    elif p['type'] == "hough":
        rho, theta = p['rho'], p['theta']
        if theta == np.pi/2:
            x_min, x_max = 0, shape[1] - 1
            p1 = np.array([rho, x_min])
            p2 = np.array([rho, x_max])
        else:
            p1 = (y_min, (rho - y_min * np.sin(theta)) / np.cos(theta))
            p2 = (y_max, (rho - y_max * np.sin(theta)) / np.cos(theta))

    else:
        raise NotImplementedError
    
    return p1, p2

def compute_line_pixels(shape, p):
    """Given a line parametrized by 'p', returns coordinates for visible line pixels (Nx2)."""
    # Compute visible endpoints of the line
    p1, p2 = line_endpoints(shape, p)

    # Compute pixels belonging to the line
    pixel_coords = line(*np.rint(p1).astype(np.int32), *np.rint(p2).astype(np.int32))
    
    # Merge coordinates in a matrix
    pixel_coords = np.stack(pixel_coords, axis=1)
    
    # Take coordinates in the 1st quadrant
    pixel_coords = pixel_coords[np.min(pixel_coords, axis=1)>=0]
    
    # Take coordinates in the visible region
    visible = (pixel_coords[:,0] < shape[0]) & (pixel_coords[:,1] < shape[1])
    pixel_coords = pixel_coords[visible]
    
    return pixel_coords

def convert_coord2mask(shape, pixel_coords):
    """Convert pixel coordinates of a shape into a binary mask"""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[pixel_coords[:, 0], pixel_coords[:, 1]] = 255
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

def apply_functions(image, func_stack, config_stack):
    show_image(image)
    for i in range(len(func_stack)):
        key = func_stack[i]
        image = image_processing.funcs[key](image, **config_stack[i])
    return image
    
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
    color = (255, 0, 0) if len(image.shape) == 3 else 255
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


def print_full(arr, outfile="temp.csv"):
    header = list(map(str, np.arange(1, arr.shape[1] + 1, 1)))
    print(list(header))
    pd.DataFrame(arr).to_csv(outfile, header=header, index=False, index_label=True)
