#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import signal
import sys
from time import time

import numpy as np
from PIL import Image
import pyrealsense2 as rs # Intel RealSense cross-platform open-source API

RUNNING = True

def validate_args_early(args):
    try:
        args['first'] = int(args['first'])
        assert args['first'] > 0, f"first({args['first']}) has to be positive"
        
        if "inf" in args['last']:
            args['last'] = float(args['last'])
        else:
            args['last'] = int(args['last'])

        assert args['first'] <= args['last'], f"first({args['first']}) cannot be larger than last({args['last']})"
        
        args['step'] = int(args['step'])
        
        if "inf" in args['max']:
            args['max'] = float(args['max'])
        else:
            args['max'] = int(args['max'])
    
    except Exception as e:
        print("Error:", e)
        exit(-1)

def validate_args_late(args, frame_count):
    try:
        assert args['first'] <= frame_count, f"first({args['first']}) cannot be larger than max frames({frame_count}) in the input file"
        
        # Set last to frame_count if it exceeds frame_count at the EOF
        if args['last'] > frame_count:
            # print(f"last({args['last']}) updated with frame count({frame_count})")
            args['last'] = frame_count

        # If step flag is not given but max flag is provided, then sample max items with equal intervals between from first to last
        if '--step' not in sys.argv and args['max'] != np.inf:
            step = (args['last'] - args['first'] + 1) // args['max']
            args['step'] = step if step > 0 else 1 # step should be at least 1
            # print(f'Step size is dynamically adjusted to {args["step"]} with respect to start: {args["first"]}, last: {args["last"]}, max: {args["max"]} parameters')

    except Exception as e:
        print("Error:", e)
        exit(-1)

def interrupt_handler(signum, frame):
    global RUNNING
    RUNNING = not RUNNING
    if RUNNING:
        print("\nResuming execution.")
    else:
        print("\nTerminating.")

# Handles command line arguments
def arg_handler():
    parser = argparse.ArgumentParser(description='Extract information from rosbag files', 
                                     add_help=True)
    # Optional flags
    optional = parser._optionals
    optional.title = 'Optional arguments'
    optional._option_string_actions["-h"].help = "Show this help message and exit"

    optional.add_argument("--first",
                       help="Specify the first frame number (default: 1)",
                       metavar="FIRST",
                       default=1)

    optional.add_argument("--last",
                       help="Specify the last frame number (default: inf)",
                       metavar="LAST",
                       default="inf")

    optional.add_argument("--step",
                       help="Specify the increment at each iteration (default: 1)",
                       metavar="STEP",
                       default=1)

    optional.add_argument("--max",
                       help="Specify maximum frames to be extracted (default: inf)",
                       metavar="MAX",
                       default="inf")
    
    optional.add_argument("--out",
                       type=str,
                       help="Specify output folder name (default: './extracted')", 
                       metavar="FOLDER",
                       default="./extracted")

    optional.add_argument("--depth", help="Save depth as PNG", default=False, action="store_true")

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

def calculate_frame_count(duration, fps):
    seconds = duration.total_seconds() # duration: datetime.timedelta -> seconds: float
    frame_count = round(seconds * fps)
    print(f"Duration: {seconds} secs, FPS: {fps}, Expected Number of Frames: ~{frame_count}\n")
    return frame_count

def main():
    signal.signal(signal.SIGINT, interrupt_handler)

    args = vars(arg_handler())

    validate_args_early(args)
    
    print("\nFlags:")
    for k, v in args.items():
        print("- {}: {}".format(k, v))
    print()

    # Create directory (if not already exists)
    try:
        os.mkdir(args["out"])
    except FileExistsError:
        answer = input("WARNING: {} exists. Would you like to continue anyway? [y/N]: ".format(args["out"]))
        if answer.lower() != 'y':
            print("Exited.")
            exit(-1)

    # Setup:
    pipe = pipe_profile = frame_count = None
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
        frame_count = calculate_frame_count(timedelta, fps)
        
        end_time = time()

        print(f"Bag file is successfully read in {end_time-start_time} secs\n")
    
    except RuntimeError as e:
        print("Error:", e)
        exit(-1)

    validate_args_late(args, frame_count)

    filename = "{name}_{type}.{ext}"
    first = args["first"]
    last = args["last"]
    step = args["step"]
    max = args["max"]
    align = rs.align(rs.stream.color) # Create alignment primitive with color as its target stream
    colorizer = rs.colorizer()
    frame_time_map = {}
    frameset = None
    iter = extracted = 0
    
    start_time = time()
    while extracted < max and RUNNING:
        # Read a frame
        try:
            frameset = pipe.wait_for_frames(timeout_ms=5000)
        
        # End if no more frames to read
        except RuntimeError:
            print("End of file reached, terminating.")
            break

        iter += 1

        # Discard current frame if start point or step not reached
        if iter < first or (iter - first) % step != 0:
            continue

        # End if end point reached
        elif iter > last:
            print(f"Last iteration reached, terminating.")
            break

        extracted += 1

        color_frame = frameset.get_color_frame()
        # depth_frame = frameset.get_depth_frame() # Unaligned depth data

        # Extract metadata identifying the frame
        frame_id = frameset.get_frame_number()
        frame_stamp = frameset.get_timestamp()
        frame_time_map[frame_id] = frame_stamp

        # Print timestamp for current frame
        print("Iteration: {}, Frame: {}, Frame ID: {}, Capture Time: {}".format(iter, extracted, frame_id, frame_stamp))

        # Store RGB Data
        color = np.asanyarray(color_frame.get_data())
        image = Image.fromarray(color)
        rgb_name = filename.format(name=frame_id, type="rgb", ext="png")
        image.save(os.path.join(args["out"], rgb_name))

        # Stream Alignment (depth to color)
        frameset = align.process(frameset) # Now the two images are pixel-perfect aligned

        # Get aligned depth frame
        aligned_depth_frame = frameset.get_depth_frame()

        # Store depth array
        depth_arr = np.asanyarray(aligned_depth_frame.get_data())
        depth_arr = depth_arr * depth_scale # Store depth in meters
        arr_name = filename.format(name=frame_id, type="depth", ext="npy")
        np.save(os.path.join(args["out"], arr_name), depth_arr)

        # Store depth as colorized image (optional)
        if args["depth"]:
            colorized = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
            image = Image.fromarray(colorized)
            depth_rgb_name = filename.format(name=frame_id, type="depth", ext="png")
            image.save(os.path.join(args["out"], depth_rgb_name))

    # Store mapping from frame ID to timestamp
    base = os.path.basename(args["file"])
    video_id = os.path.splitext(base)[0]
    dict_name = filename.format(name=video_id, type="time", ext="json")
    with open(os.path.join(args["out"], dict_name), "w") as file:
        json.dump(frame_time_map, file, indent=4)

    # Cleanup at the end:
    pipe.stop()
    end_time = time()
    print("{} frames extracted in {} secs.".format(extracted, end_time-start_time))

if __name__ == "__main__":
    main()
