#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import signal
import sys
from time import time

import numpy as np
from PIL import Image
import pyrealsense2 as rs # Intel RealSense cross-platform open-source API

RUNNING = True

def check_skip_count(value):
    try:
        value = int(value)
        assert (value >= 0), "Non-negative integer is expected but got: {}".format(value)
    except Exception as e:
        print("ERROR:", e)
        exit(-1)
    return value

def check_frame_count(value):
    if value.lower() == 'all':
        value = np.inf
    else:
        try:
            value = int(value)
            assert (value > 0), "Positive integer is expected but got: {}".format(value)
        except Exception as e:
            print("ERROR:", e)
            exit(-1)
    return value

def interrupt_handler(signum, frame):
    global RUNNING
    RUNNING = not RUNNING
    if RUNNING:
        print("\nResuming execution.")
    else:
        print("\nTerminating.")
    return

# Handles command line arguments
def arg_handler():
    parser = argparse.ArgumentParser(description='Extract information from rosbag files', 
                                     add_help=True)
    # Optional flags
    optional = parser.add_argument_group(title='Optional arguments')

    optional.add_argument("-ns", "--nskip",
                       help="Specify the number of frames to be skipped from the beginning (default: 0)",
                       type=check_skip_count,
                       metavar="NUM_SKIP",
                       default=0)

    optional.add_argument("-nf", "--nframes",
                       help="Specify the number of frames to be extracted (positive int or 'all', default: 1)",
                       type=check_frame_count,
                       metavar="NUM_FRAME",
                       default=1)
    
    optional.add_argument("-o", "--outputfolder", 
                       type=str,
                       help="Specify output folder name (default: './extracted')", 
                       metavar="FOLDER",
                       default="./extracted")

    required = parser.add_argument_group(title='Required arguments')

    required.add_argument("-f", "--file", 
                   type=str,
                   help="Specify rosbag path (required)", 
                   metavar="PATH",
                   required=True)

    args = parser.parse_args()

    return args

def print_intrinsics(intrinsics):
    print("Intrinsics:")
    print(f"- H: {intrinsics.height}, W: {intrinsics.width}")
    print(f"- fx: {intrinsics.fx}, fy: {intrinsics.fy}")
    print(f"- Principle point in pixel coordinates: ({intrinsics.ppx}, {intrinsics.ppy})")
    print(f"- Distortion Model: {intrinsics.model}")
    print(f"- Coefficients: {intrinsics.coeffs}\n")

def calculate_frame_count(timedelta, fps):
    duration = timedelta.total_seconds() # datetime.timedelta -> float
    frame_count = int(duration * fps)
    print(f"Duration: {duration} secs, FPS: {fps}, Expected Number of Frames: ~{frame_count}\n")
    return frame_count

def main():
    signal.signal(signal.SIGINT, interrupt_handler)

    args = vars(arg_handler())
    
    print("\nFlags:")
    for k, v in args.items():
        print("- {}: {}".format(k, v))
    print()

    # Create directory (if not already exists)
    # try:
    #     os.mkdir(args["outputfolder"])
    # except FileExistsError:
    #     answer = input("WARNING: {} exists. Would you like to continue anyway? [y/N]: ".format(args["outputfolder"]))
    #     if answer.lower() != 'y':
    #         print("Exited.")
    #         exit(-1)

    # Setup:
    first_frame = 0
    pipe = pipe_profile = last_frame = None
    try:
        start_time = time()

        # Initialization & rosbag reading
        pipe = rs.pipeline() # The pipeline simplifies the user interaction with the device and computer vision processing modules.
        cfg = rs.config()    # The config allows pipeline users to request filters for the pipeline streams and device selection and configuration.
        cfg.enable_device_from_file(args["file"], repeat_playback=False) # Playback a file but do not loop over the file.
        pipe_profile = pipe.start(cfg) # The pipeline profile includes a device and a selection of active streams, with specific profiles.
        device = pipe_profile.get_device() # Includes color & depth sensors
        color_stream = pipe_profile.get_stream(rs.stream.color) # Includes fps information
        
        # Get intrinsics
        intrinsics = color_stream.as_video_stream_profile().intrinsics # Intrinsics can only be accessible via video_stream_profile.
        print_intrinsics(intrinsics)
        
        # Get scale
        depth_scale = device.first_depth_sensor().get_depth_scale()
        print(f"Depth scale: {depth_scale:.3f} meters\n")
        
        # Disable real-time reading to prevent frame drops
        playback = device.as_playback() # Cast device as playback object
        playback.set_real_time(False)
        timedelta = playback.get_duration()
        fps = color_stream.fps()
        last_frame = calculate_frame_count(timedelta, fps)
        
        end_time = time()

        print(f"Bag file is successfully read in {end_time-start_time} secs\n")
    
    except RuntimeError as e:
        print("ERROR:", e)
        exit(-1)

    count = 0 # + args["nskip"]
    max_count = args["nframes"] # + args["nskip"]
    start_time = time()
    align = rs.align(rs.stream.color) # Create alignment primitive with color as its target stream
    frame_time_map = {}
    filename = "{frame}.{ext}"
    
    # TODO: Use one of the followings later
    # colorizer = rs.colorizer()       # Create colorizer primitive
    # colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    # image = Image.fromarray(colorized_depth)
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    while RUNNING:
        # Store next frameset for later processing:
        try:
            frameset = pipe.wait_for_frames()
        except RuntimeError:
            break

        count += 1

        color_frame = frameset.get_color_frame()
        # depth_frame = frameset.get_depth_frame() # (Unaligned) Depth Data

        # Extract metadata identifying the frame
        frame_number = frameset.get_frame_number()
        frame_stamp = frameset.get_timestamp()
        frame_time_map[frame_number] = frame_stamp

        # Print timestamp for current frame
        print("Frame: {}, Frame ID: {}, Capture Time: {}".format(count, frame_number, frame_stamp))

        ## RGB Data
        color = np.asanyarray(color_frame.get_data())
        image = Image.fromarray(color)
        # filename.format(frame=frame_number, ext="png")
        # image.save(os.path.join(args["outputfolder"], str(count) + "_rgb.png"))

        # Stream Alignment (depth to color)
        frameset = align.process(frameset) # Now the two images are pixel-perfect aligned and you can use depth data just like you would any of the other channels.

        # Get aligned depth frame
        aligned_depth_frame = frameset.get_depth_frame()

        # Depth array
        depth_arr = np.asanyarray(aligned_depth_frame.get_data())
        depth_arr = depth_arr * depth_scale # Store depth in meters
        # filename.format(frame=frame_number, ext="npy")
        # np.save(os.path.join(args["outputfolder"], str(count) + "_depth.npy"), depth_arr)

    # Cleanup at the end:
    pipe.stop()
    end_time = time()
    print("{} frames extracted in {} secs.".format(count - args["nskip"], end_time-start_time))

if __name__ == "__main__":
    main()
