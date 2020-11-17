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

    optional.add_argument("-d", "--depth", help="Save depth as PNG", default=False, action="store_true")
    
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

def stream_length(pipe):
    frame = pipe.wait_for_frames()
    first = previous = frame.get_frame_number()
    timestamp = frame.get_timestamp()
    count = 1
    print("Frame: {}, Frame ID: {}, Capture Time: {}".format(count, first, timestamp))
    while True:
        frame = pipe.wait_for_frames()
        current = frame.get_frame_number()
        timestamp = frame.get_timestamp()
        if current == first or current == previous:
            return count
        count += 1
        print("Frame: {}, Frame ID: {}, Capture Time: {}".format(count, current, timestamp))

def main():
    signal.signal(signal.SIGINT, interrupt_handler)

    args = vars(arg_handler())
    
    print("\nFlags:")
    for k, v in args.items():
        print("- {}: {}".format(k, v))
    print()

    try:
        os.mkdir(args["outputfolder"])
    except FileExistsError:
        answer = input("WARNING: {} exists. Would you like to continue anyway? [y/N]: ".format(args["outputfolder"]))
        if answer.lower() != 'y':
            print("Exited.")
            exit(-1)

    # Setup:
    pipe = profile = None
    try:
        start_time = time()
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device_from_file(args["file"])
        profile = pipe.start(cfg)
        end_time = time()
        print("Bag file is successfully read in {}s".format(end_time-start_time))
    except RuntimeError as e:
        print("ERROR:", e)
        exit(-1)

    # Skip ns first frames to give the auto-exposure time to adjust
    for i in range(args["nskip"]):
        print("Skipping frame:", i)
        pipe.wait_for_frames()

    count = 0
    first_frame = previous_frame = None
    start_time = time()
    while count < args["nframes"] and RUNNING:
        # Store next frameset for later processing:
        frameset = pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        # Extract metadata identifying the frame
        frame_number = frameset.get_frame_number()
        frame_stamp = frameset.get_timestamp() # TODO: Save timestamp later

        # Exit loop if stream restarts
        if frame_number == first_frame or frame_number == previous_frame:
            break

        # Update loop parameters for the next iteration
        count += 1
        previous_frame = frame_number

        # Remember first frame
        if count == 1:
            first_frame = frame_number

        # Print timestamp for current frame
        print("Frame: {}, Frame ID: {}, Capture Time: {}".format(count, frame_number, frame_stamp))

        ## RGB Data
        color = np.asanyarray(color_frame.get_data())
        image = Image.fromarray(color)
        image.save(os.path.join(args["outputfolder"], str(count) + "_rgb.png"))

        ## Depth Data
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        ## Stream Alignment (depth to color)

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame() # Now the two images are pixel-perfect aligned and you can use depth data just like you would any of the other channels.

        # Depth image
        if args["depth"]:
            colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
            image = Image.fromarray(colorized_depth)
            image.save(os.path.join(args["outputfolder"], str(count) + "_depth.png"))

        # Depth array
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_arr = np.asanyarray(aligned_depth_frame.get_data()) # in meters
        depth_arr = depth_arr * depth_scale
        np.save(os.path.join(args["outputfolder"], str(count) + "_depth.npy"), depth_arr)

    # Cleanup at the end:
    pipe.stop()
    end_time = time()
    print("{} frames extracted in {}s.".format(count, end_time-start_time))

if __name__ == "__main__":
    main()
