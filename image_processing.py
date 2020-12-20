## Image Processing Functions
# Define image processing functions to be used here.

import cv2
from cyvlfeat.sift import sift, dsift, phow
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image

import utils

def convert_grayscale(image, visualize=True, **kwargs): # **kwargs is ignored
    """Takes an RGB (3 channel) image, returns the grayscale image (1 channel)"""
    print("Converting to grayscale")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def enhance_contrast(image, visualize=True, **kwargs): # **kwargs is ignored
    print("Enhancing contrast")
    image = cv2.equalizeHist(image)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_threshold(gray_img, visualize=True, **kwargs):
    """Takes grayscale image, returns thresholded version."""
    print("Applying standard threshold with args:", kwargs)
    _retval, gray_img = cv2.threshold(gray_img, **kwargs)
    if visualize:
        utils.show_image(gray_img, cmap="gray")
    return gray_img

def apply_channelwise_threshold(image, visualize=True, **kwargs):
    """Takes multi-channel image, applies different threshold per channel, returns thresholded version."""
    print("Applying channelwise threshold with args:", kwargs)
    C = image.shape[2]
    max_vals = np.array(kwargs.get("max_vals", (255,)*C))
    min_vals = np.array(kwargs.get("min_vals", (0,)*C))
    assert len(max_vals) == len(min_vals)
    assert len(max_vals) == C
    assert np.all(0 <= min_vals) 
    assert np.all(max_vals >= min_vals) 
    assert np.all(max_vals <= 255)

    mask = (min_vals <= image) & (image <= max_vals)            # Apply two ends of threshold for each channel individually
    mask = np.bitwise_and.reduce(mask, axis=2)[..., np.newaxis] # Reduce binary masks with C-channels into 1-channel by taking AND over channels
    thresh_image = image * mask
    if visualize:
        utils.show_image(thresh_image, cmap="gray")
    return thresh_image

def apply_adaptive_threshold(gray_img, **kwargs):
    """Takes grayscale image, returns thresholded version."""
    print("Applying adaptive threshold with args:", kwargs)
    return cv2.adaptiveThreshold(gray_img, **kwargs)

def apply_spatial_threshold(image, visualize=True, **kwargs):
    """Takes an image, crop certain region, set rest to zero."""
    print("Applying spatial threshold with args:", kwargs)
    
    h, w = image.shape[0:2]
    
    range_x = kwargs.get("range_x", (w, 0))
    range_y = kwargs.get("range_y", (0, h))
    
    assert range_x[0] <= range_x[1],\
        "min_x({}) cannot be greater than max_x({})".format(*range_x)
    
    assert range_x[1] <= w,\
        "Input: {}, Image Width: {}".format(range_x[1], w)
    
    assert range_x[0] >= 0 and range_x[1] >= 0,\
        "Input: {} must contain positive integer".format(range_x)
    
    assert range_y[0] <= range_y[1],\
        "min_y({}) cannot be greater than max_y({})".format(*range_y)
    
    assert range_y[1] <= h,\
        "Input: {}, Image Height: {}".format(range_y[1], h)
    
    assert range_y[0] >= 0 and range_y[1] >= 0,\
        "Input: {} must contain positive integer".format(range_y)
        
    new_image = np.zeros_like(image)
    crop = image[range_y[0]:range_y[1], range_x[0]:range_x[1]]
    new_image[range_y[0]:range_y[1], range_x[0]:range_x[1]] = crop
    
    if visualize:
        utils.show_image(new_image)
    
    return new_image

def apply_roi_threshold(image, visualize=True, **kwargs):
    """Mask out ROI, return image as well as pixel coordinates for ROI corners"""
    print("Applying ROI threshold with args:", kwargs)
    C = image.shape[2] if len(image.shape) == 3 else 1
    white = (255,) * C
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, kwargs["roi_corners"], color=white)
    image = cv2.bitwise_and(image, mask)

    if visualize:
        utils.show_image(image)
        
    return image

def apply_filter(image, kernel, visualize=True):
    print("Applying filter:")
    print(kernel)
    image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_avg_blur(image, visualize=True, **kwargs):
    print("Applying average blur with args:", kwargs)
    image = cv2.blur(image, **kwargs)
    if visualize:
        utils.show_image(image)
    return image

def apply_median_blur(image, visualize=True, **kwargs):
    print("Applying median blur with args:", kwargs)
    image = cv2.medianBlur(image, **kwargs)
    if visualize:
        utils.show_image(image)
    return image

def apply_bilateral_filter(image, visualize=True, **kwargs):
    print("Applying bilateral filter with args:", kwargs)
    image = cv2.bilateralFilter(image, **kwargs)
    if visualize:
        utils.show_image(image)
    return image

def apply_gaussian_blur(image, visualize=True, **kwargs):
    print("Applying Gaussian blur with args:", kwargs)
    image = cv2.GaussianBlur(image, **kwargs)
    if visualize:
        utils.show_image(image)
    return image

def apply_laplacian(image, visualize=True, **kwargs): 
    print("Applying Laplacian filter for vertical and horizontal edges with args:", kwargs)
    image = cv2.Laplacian(image, cv2.CV_64F, **kwargs) # Detect both + & - edges
    image = np.absolute(image) # Take absolute value to make it unsigned
    image = np.uint8(image)    # Convert dtype back to uint8
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_sobelv(image, visualize=True, **kwargs):
    print("Applying Sobel filter for vertical edges (along y-axis) with args:", kwargs)
    image = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, **kwargs) # Detect both + & - edges
    image = np.absolute(image) # Take absolute value to make it unsigned
    image = np.uint8(image)    # Convert dtype back to uint8
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_sobelh(image, visualize=True, **kwargs):
    print("Applying Sobel filter for horizontal edges (along x-axis) with args:", kwargs)
    image = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, **kwargs) # Detect both + & - edges
    image = np.absolute(image) # Take absolute value to make it unsigned
    image = np.uint8(image)    # Convert dtype back to uint8
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_scharrh(image, visualize=True, **kwargs):
    print("Applying Scharr filter for horizontal edges (along x-axis) with args:", kwargs)
    image = cv2.Scharr(image, cv2.CV_64F, dx=0, dy=1, **kwargs)
    image = np.absolute(image) # Take absolute value to make it unsigned
    image = np.uint8(image)    # Convert dtype back to uint8
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_scharrv(image, visualize=True, **kwargs):
    print("Applying Scharr filter for vertical edges (along y-axis) with args:", kwargs)
    image = cv2.Scharr(image, cv2.CV_64F, dx=1, dy=0, **kwargs) # Detect both + & - edges
    image = np.absolute(image) # Take absolute value to make it unsigned
    image = np.uint8(image)    # Convert dtype back to uint8
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_prewitth(image, visualize=True, **kwargs):
    print("Applying Prewitt filter for horizontal edges (along x-axis) with args:", kwargs)
    image = filters.prewitt_h(image)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_prewittv(image, visualize=True, **kwargs):
    print("Applying Prewitt filter for vertical edges (along y-axis) with args:", kwargs)
    image = filters.prewitt_v(image)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_farid(image, visualize=True, **kwargs):
    print("Applying Farid filter with args:", kwargs)
    image = filters.farid(image)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_faridh(image, visualize=True, **kwargs):
    print("Applying Farid filter for horizontal edges (along x-axis) with args:", kwargs)
    image = filters.farid_h(image)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_faridv(image, visualize=True, **kwargs):
    print("Applying Farid filter for vertical edges (along y-axis) with args:", kwargs)
    image = filters.farid_v(image)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_roberts(image, visualize=True, **kwargs):
    print("Applying Roberts filter for diagonal edges with args:", kwargs)
    image = filters.roberts(image)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_canny(image, visualize=True, **kwargs):
    """Takes an RGB or grayscale image, returns the binary mask for edge locations"""
    print("Applying canny with args:", kwargs)
    image = cv2.Canny(image, **kwargs)
    if visualize:
        utils.show_image(image, cmap="gray")
    return image

def apply_shi_tomasi(gray_image, visualize=True, **kwargs):
    corners = cv2.goodFeaturesToTrack(gray_image, **kwargs) # maxCorners=49, qualityLevel=0.1, minDistance=20
    corners = cv2.cornerSubPix(gray_image, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.01))
    corners = np.squeeze(corners, axis=1) # (N, 1, 2) -> (N, 2)
    corners[:, [0,1]] = corners[:, [1,0]] # (x, y) -> (row/y, col/x)
    if visualize:
        utils.show_corners(gray_image, corners)
    return corners

def apply_dsift(gray_image, visualize=True, top_n=100, **kwargs):
    corners, descr = dsift(gray_image, **kwargs)
    descr_norm = np.linalg.norm(descr, ord=2, axis=1)
    top_corners = np.argsort(descr_norm)[-top_n:]
    corners = corners[top_corners]
    if visualize:
        utils.show_corners(gray_image, corners)
    return corners

def apply_sift(gray_image, visualize=True, **kwargs):
    corners, descr = sift(gray_image, compute_descriptor=True, **kwargs)
    # TODO: Enhance
    # descr_norm = np.linalg.norm(descr, ord=2, axis=1)
    # top_corners = np.argsort(descr_norm)[-top_n:]
    # corners = corners[top_corners]
    if visualize:
        utils.show_corners(gray_image, corners)
    return corners

def apply_convex_hull(points, shape, visualize=True, **kwargs):
    """Given a point set, return the vertices of convex polygon contanining it."""
    points[:,[0,1]] = points[:,[1,0]]       # (row/y, col/x) -> (x, y)
    vertices = cv2.convexHull(points)
    vertices = np.squeeze(vertices, axis=1) # (N, 1, 2) -> (N, 2)
    vertices[:, [0,1]] = vertices[:, [1,0]] # (x, y) -> (y/row, x/col)

    if visualize:
        utils.show_shapes(np.zeros(shape, dtype=np.uint8), [np.rint(vertices).astype(np.int32)], cmap="gray")

    return vertices

def apply_contour(bin_image, visualize=True, **kwargs):
    """
    Takes binary image (obtained with Canny or thresholding) to find contours (which separate white region from black background).
    'contours' is a list of all the contours in the image. 
    Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    Returns 'contours'.
    """
    print("Applying contours with args:", kwargs)
    _bin_image, contours, _hierarchy = cv2.findContours(bin_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        contours[i] = np.squeeze(contours[i], axis=1) # (N, 1, 2) -> (N, 2)
        contours[i] = contours[i][:, [1,0]] # (x, y) -> (y/row, x/col)
    
    if visualize:
        utils.show_shapes(np.zeros_like(bin_image), contours, shapeIdx=-1, cmap="gray")
    
    return contours

def apply_hough_line(bin_image, visualize=True, print_lines=20, plot_hough_space=True, **kwargs):
    """
    Takes binary image (obtained with Canny or thresholding) to find parameters of lines that exist in the image.
    'peak_params' is a list of parameters for each peak denoting a different line in the image. 
    Each individual peak parameter is a NumPy array of (votes, theta, rho).
    Returns pixel coordinates for all points of each line (line_list).
    """
    print("Applying Hough line with args:", kwargs)

    num_samples = 180 * kwargs["theta_res"]
    test_angles = np.linspace(0, np.pi, num_samples)
    
    votes, thetas, rhos = hough_line(bin_image, theta=test_angles)
    print(f"[Info]: Max vote is {votes.max()}")
    peak_params = hough_line_peaks(votes, thetas, rhos,
                                   num_peaks=kwargs["num_lines"],
                                   threshold=None if kwargs["adaptive_thresh"] else kwargs["threshold"],
                                   min_distance=kwargs["min_distance"], 
                                   min_angle=kwargs["min_angle"])
    
    # (votes: np.array, thetas: np.array, rhos: np.array) -> np.array([[vote0, theta0, rho0], ...)
    peak_params = np.stack(peak_params, axis=1)

    # Apply rho and theta filters to the results
    extra_filters = (np.deg2rad(kwargs["min_theta"]) <= peak_params[:, 1]) & (np.deg2rad(peak_params[:, 1]) <= kwargs["max_theta"])
    extra_filters &= (kwargs["min_rho"] <= peak_params[:, 2]) & (peak_params[:, 2] <= kwargs["max_rho"])
    peak_params = peak_params[extra_filters]
    num_lines = peak_params.shape[0]

    # Visualize detected line as points in the Hough space
    if plot_hough_space:
        print("[Info]: Visualizing Hough space.")
        
        plt.figure(figsize=(10,10))
        plt.title('Hough Space')
        plt.xlabel('Angles (degrees)')
        plt.ylabel('Distance (pixels)')
        
        # Visualize votes
        plt.imshow(np.log(1 + votes),
                   extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]],
                   cmap="gray")
        plt.grid(color='r', linestyle='--', linewidth=1)
        plt.axis('auto')

        # Mark detected peaks
        t = np.rad2deg(peak_params[:, 1])
        r = peak_params[:, 2]
        plt.scatter(t, r, color='blue', marker='x')
        for idx in range(len(r)):
            plt.annotate(idx+1, (t[idx], r[idx]), 
                         fontsize="large", 
                         color="blue",
                         textcoords="offset pixels", 
                         xytext=(-5,5))
        plt.show()
    
    # Print parameters of detected lines
    if print_lines > 0:
        print_lines = num_lines if print_lines > num_lines else print_lines
        print(f"[Info]: {num_lines} lines detected. Showing results for ({print_lines}/{num_lines}):")
        for i, (vote, theta, rho) in enumerate(peak_params[:print_lines], 1):
            print(f"- Line {i}, vote: {vote}, rho: {rho}, theta: {np.rad2deg(theta)}")
    
    # Consider a single line obtained by averaging results
    if kwargs["average"] and num_lines > 0:
        print("[Info]: Averaging results to obtain a single line.")
        peak_params = np.mean(peak_params, axis=0, keepdims=True)
        vote, theta, rho = peak_params[0]
        print(f"- Averaged line, vote: {vote}, rho: {rho}, theta: {np.rad2deg(theta)}")
    
    # Compute pixel coordinates for each line
    line_list = []
    for vote, theta, rho in peak_params:
        pixel_coords = utils.compute_line_pixels(bin_image.shape, rho, theta)
        line_list.append(pixel_coords)
    
    # Visualize binary mask for all lines found
    if visualize:
        utils.show_shapes(np.zeros_like(bin_image, dtype=np.uint8), line_list, shapeIdx="all", cmap="gray")
        
    return line_list

# Binding names to actual methods:
preproc_funcs = {
    "Grayscale": convert_grayscale,
    "Contrast": enhance_contrast,
    "Threshold": apply_threshold,
    "Adaptive Threshold": apply_adaptive_threshold,
    "Spatial Threshold": apply_spatial_threshold,
    "ROI Threshold": apply_roi_threshold,
    "Average Blur": apply_avg_blur,
    "Median Blur": apply_median_blur,
    "Bilateral Filter": apply_bilateral_filter,
    "Gaussian Blur": apply_gaussian_blur,  
    "Laplacian": apply_laplacian,
    "Sobel Vertical": apply_sobelv,
    "Sobel Horizontal": apply_sobelh,
}

detect_funcs = {
    "Canny Edges": apply_canny,
    "Contour": apply_contour,
    "Shi Tomasi": apply_shi_tomasi,
    "Dense SIFT": apply_dsift,
    "SIFT": apply_sift,
    "Hough Line": apply_hough_line,
}

funcs = {**preproc_funcs, **detect_funcs}