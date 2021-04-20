## Helper Functions
# Define general purpose methods to here.

import cv2
from matplotlib import pyplot as plt
from matplotlib import colors  
import numpy as np

def topk(arr, k, largest=True):
    """Find top k elements in D dimensional array and return values (k,) and indices (kxD)"""
    assert k > 0, "k({}) has to be positive.".format(k)
    flat_arr = np.ravel(arr)
    # np.argsort sorts in ascending order, take last n elements in reverse order
    topk = np.arange(k)
    if largest:
        topk = -(topk + 1)
    flat_top_idxs = np.argsort(flat_arr)[topk]
    top_idxs = np.unravel_index(flat_top_idxs, arr.shape)
    top_vals = arr[top_idxs]
    return (top_vals, np.array(list(zip(*top_idxs))))

def create_template(size, position=4):
    """
    Used for defining template required for apply_template_matching.
    Pixels within the position are set in the template.
    Convention for position:
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

def estimate_y(depth, m=0.0027, b=0.06):
    """
    Given depth in meters, compute an approximation 
    to y coordinates in pixels based on a predefined linear model.
    """
    return (1 / depth - b) / m

def estimate_depth(y, m=0.0027, b=0.06):
    """
    Given y coordinates in pixels, compute an approximation 
    to depth in meters based on a predefined linear model.
    """
    return 1 / (m * y + b)

def estimate_dy(depth0, ddepth, m=0.0027, b=0.06):
    """
    Given depth difference in meters, compute an approximation 
    to difference along y axis in pixels based on a predefined linear model.
    """
    y0 = estimate_y(depth0, m, b)
    depth1 = depth0 + ddepth
    y1 = estimate_y(depth1, m, b)
    return abs(y1-y0)

def estimate_ddepth(depth0, dy, m=0.0027, b=0.06):
    """
    Given difference along y axis in pixels, compute an approximation 
    to depth difference in meters based on a predefined linear model.
    """
    y0 = estimate_y(depth0, m, b)
    y1 = y0 - dy
    depth1 = estimate_depth(y1, m, b)
    return abs(depth1-depth0)

def parabola(y, a, b, c):
    """2nd degree parabola formula."""
    return (a * y**2) + (b * y) + c

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

def coord_to_mask(shape, yx, thickness=1):
    """Convert given pixel coordinates of a mask into a binary mask image of given shape."""
    shape = shape[:2]
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Empty coordinate array means there is no edge pixel, return a blank mask in this case.
    if len(yx) == 0:
        return mask

    # When there are edge pixels, mark them on the mask.
    xy = yx[:, [1,0]]
    mask = cv2.polylines(mask, [xy], False, 255, thickness, cv2.LINE_8)
    mask = (mask.astype(np.bool) * 255).astype(np.uint8)
    return mask

def overlay_coord(image, yx, thickness=1, color='springgreen'):
    """Given edge pixel coordinates overlay edge mask on RGB image."""
    overlaid = image.copy()
    # Empty coordinate array means there is no edge pixel, do nothing
    if len(yx) == 0:
        return overlaid

    # When there are edge pixels, mark them on the mask.
    xy = yx[:, [1,0]]
    color = [255 * x for x in colors.to_rgb(color)] if len(image.shape) == 3 else 255
    overlaid = cv2.polylines(overlaid, [xy], False, color, thickness, cv2.LINE_8)
    return overlaid

def overlay_mask(image, mask, color='springgreen'):
    """Overlay given edge mask on RGB image. Image and mask have to be in [0,1]."""
    overlaid = image.copy() / 255
    color = colors.to_rgb(color) if len(image.shape) == 3 else 1
    alpha_fg = mask[:,:,np.newaxis]
    alpha_bg = 1 - alpha_fg
    overlaid = alpha_bg * overlaid + alpha_fg * np.array(color)
    return np.clip(np.array(overlaid), 0, 1)

def prepare_show_image(image, h=10, w=10, cmap=None):
    # Prepare plot, but do not show.
    plt.figure(figsize=(w,h))
    plt.imshow(image, cmap=cmap)
    # plt.grid(True)
    # plt.xticks(np.arange(0, image.shape[1], 40))

def show_image(image, h=10, w=10, cmap=None, ticks=True):
    # Prepare and show plot.
    prepare_show_image(image, h, w, cmap)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_image_pairs(left, right, h=15, w=15):
    f, ax = plt.subplots(1,2)
    f.set_figheight(h)
    f.set_figwidth(w)
    ax[0].imshow(left)
    ax[1].imshow(right)
    plt.show()

def generate_edge(depth_arr, edge1, intrinsics, shift3D=[-0.25, 0, 0]):
    """
    Given a translation vector in 3D in meters, compute pixel coordinates for the new edge.
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
    depths = depth_arr[edge1[:,0], edge1[:,1]]
    dx, dy, dz = shift3D
    fx, fy = intrinsics.fx, intrinsics.fy
    depths += dz
    dx = fx * dx / depths
    dy = fy * dy / depths
    edge2 = np.rint(edge1 + np.c_[dy, dx]).astype(np.int32)
    return edge2

def generate_lane_pixels(edge1, edge2, top_offset=0, bot_offset=1, num_lane=15):
    """
    Given pixels coordinates for two edges of the road, compute pixel coordinates for artificial lanes in between.
    
    Parameters
    ----------
    edge1 : ndarray
        Pixel coordinates for the first edge of the road, with shape Nx2
    edge2 : ndarray
        Pixel coordinates for the second edge of the road, with shape Nx2
    top_offset : int
        Offset from the top for drawing lane in pixels along y axis
    bot_offset : int
        Offset from the bottom for drawing lane in pixels along y axis
    num_lane : int
        Number of lanes to fit on the road

    Returns 
    -------
    lanes : ndarray
        Pixel coordinates of each lane, with shape Mx2
    """
    mid = np.rint((edge1 + edge2) / 2).astype(np.int32)
    length = slice(top_offset, -bot_offset) if bot_offset else slice(top_offset, None)
    mid = mid[length]
    lanes = np.array_split(mid, num_lane)
    return lanes[::2]

def project(cam_coords, intrinsics):
    # Unused but might be useful later.
    z = cam_coords[:, 2]
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy
    x = np.rint(cam_coords[:,0] * fx / z + ppx).astype(np.int32)
    y = np.rint(cam_coords[:,1] * fy / z + ppy).astype(np.int32)
    return np.c_[y, x]

def backproject(depth_arr, pixel_coords, intrinsics):
    # Unused but might be useful later.
    z = depth_arr[pixel_coords[:,0], pixel_coords[:,1]]
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy
    x = (pixel_coords[:,1] - ppx) / fx * z
    y = (pixel_coords[:,0] - ppy) / fy * z
    return np.c_[x, y, z]

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