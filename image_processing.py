## Image Processing Functions
# Define image processing functions to be used here.

import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import haar_like_feature_coord
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

def apply_filter(image, kernel):
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)

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
        utils.show_image(image, cmap=plt.cm.gray)
    return image

def apply_scharrv(image, visualize=True, **kwargs):
    print("Applying Scharr filter for vertical edges (along y-axis) with args:", kwargs)
    image = cv2.Scharr(image, cv2.CV_64F, dx=1, dy=0, **kwargs) # Detect both + & - edges
    image = np.absolute(image) # Take absolute value to make it unsigned
    image = np.uint8(image)    # Convert dtype back to uint8
    if visualize:
        utils.show_image(image, cmap=plt.cm.gray)
    return image

def apply_prewitth(image, visualize=True, **kwargs):
    print("Applying Prewitt filter for horizontal edges (along x-axis) with args:", kwargs)
    image = filters.prewitt_h(image)
    if visualize:
        utils.show_image(image, cmap=plt.cm.gray)
    return image

def apply_prewittv(image, visualize=True, **kwargs):
    print("Applying Prewitt filter for vertical edges (along y-axis) with args:", kwargs)
    image = filters.prewitt_v(image)
    if visualize:
        utils.show_image(image, cmap=plt.cm.gray)
    return image

def apply_farid(image, visualize=True, **kwargs):
    print("Applying Farid filter with args:", kwargs)
    image = filters.farid(image)
    if visualize:
        utils.show_image(image, cmap=plt.cm.gray)
    return image

def apply_faridh(image, visualize=True, **kwargs):
    print("Applying Farid filter for horizontal edges (along x-axis) with args:", kwargs)
    image = filters.farid_h(image)
    if visualize:
        utils.show_image(image, cmap=plt.cm.gray)
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

def apply_contour(bin_image, visualize=True, **kwargs):
    """
    Takes binary image (obtained with Canny or thresholding) to find contours (which separate white region from black background).
    'contours' is a list of all the contours in the image. 
    Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    Returns 'contours'.
    """
    print("Applying contours with args:", kwargs)
    _bin_image, contours, _hierarchy = cv2.findContours(bin_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    
    if visualize:
        utils.show_shapes(np.zeros_like(image), contours, shapeIdx=-1, cmap="gray")
    
    return contours

def apply_hough_line(bin_image, visualize=True, print_lines=20, plot_hough_space=True, **kwargs):
    """
    Takes binary image (obtained with Canny or thresholding) to find parameters of lines that exist in the image.
    'peak_params' is a list of parameters for each peak denoting a different line in the image. 
    Each individual peak parameter is a NumPy array of (votes, theta, rho).
    Returns pixel coordinates for all points of each line (line_list).
    """
    print("Applying Hough line with args:", kwargs)

    num_samples = np.abs(kwargs["max_theta"] - kwargs["min_theta"]) * kwargs["resolution"]
    test_angles = np.linspace(np.deg2rad(kwargs["min_theta"]), np.deg2rad(kwargs["max_theta"]), num_samples)
    
    votes, theta, rho = hough_line(bin_image, theta=test_angles)
    print(f"[Info]: Max vote is {votes.max()}")
    peak_params = hough_line_peaks(votes, theta, rho,
                                   num_peaks=kwargs["num_lines"],
                                   threshold=None if kwargs["adaptive_thresh"] else kwargs["threshold"],
                                   min_distance=kwargs["min_distance"], 
                                   min_angle=kwargs["min_angle"])
    
    # Visualize Hough space
    if plot_hough_space:
        print("[Info]: Visualizing Hough space.")
        plt.figure(figsize=(10,10))
        plt.imshow(np.log(1 + votes),
                   extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]), rho[0], rho[-1]],
                   cmap="gray")
        plt.title('Hough Space')
        plt.xlabel('Angles (degrees)')
        plt.ylabel('Distance (pixels)')
        plt.axis('auto')
        plt.grid(color='r', linestyle='--', linewidth=1)
        plt.show()
    
    # (votes: np.array, thetas: np.array, rhos: np.array) -> np.array((vote0, theta0, rho0), ...)
    peak_params = np.array(list(zip(*peak_params)))
    num_lines = len(peak_params)
    
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
    
    # Visualize all lines found
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
    "Hough Line": apply_hough_line,
}

funcs = {**preproc_funcs, **detect_funcs}