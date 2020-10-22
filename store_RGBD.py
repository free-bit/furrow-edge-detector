#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image
import pyrealsense2 as rs # Intel RealSense cross-platform open-source API

NUM_FRAME = 5 # TODO: Get as arg

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("./data/20201022_114416.bag") # TODO: Get as arg
profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
  pipe.wait_for_frames()

for i in range(NUM_FRAME):
	# Store next frameset for later processing:
	frameset = pipe.wait_for_frames()
	color_frame = frameset.get_color_frame()
	depth_frame = frameset.get_depth_frame()

	# Print timestamp for current frame
	print("Frame: {}, Capture Time: {}".format(i, frameset.get_timestamp()))

	## RGB Data Visualization
	color = np.asanyarray(color_frame.get_data())
	image = Image.fromarray(color)
	image.save(str(i) + "_rgb.png")

	## Depth Data Visualization
	colorizer = rs.colorizer()
	colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

	## Stream Alignment (depth to color)

	# Create alignment primitive with color as its target stream:
	align = rs.align(rs.stream.color)
	frameset = align.process(frameset)

	# Update color and depth frames:
	aligned_depth_frame = frameset.get_depth_frame()
	colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

	# Now the two images are pixel-perfect aligned and you can use depth data just like you would any of the other channels.
	image = Image.fromarray(colorized_depth)
	image.save(str(i) + "_depth.png")

	# Depth array
	depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
	depth_arr = np.asanyarray(aligned_depth_frame.get_data()) # in meters
	depth_arr = depth_arr * depth_scale
	np.save(str(i) + "_depth.npy", depth_arr)

# Cleanup:
pipe.stop()
print("Frames Captured")