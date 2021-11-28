import argparse
import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('file', help="png/jpg to gamut plot")
parser.add_argument('out', help="png/jpg to gamut plot", default="colors.jpg")
args = parser.parse_args()

im = Image.open(args.file, 'r')
pixel_values = np.array(im.getdata())[:, :3].astype(np.float64)

DARK_THRESH = 30 # take out dark pixels, where color doesn't really matter.

"""
Pixel values are in format, (r, g, b)

We need to be able to map them to their color gamut location in a NxN grid with center, (0, 0).

If we treat R, G, B as vectors, the location is simply found by r * R + g * G + b * B, where (r, g, b) is normalized.
"""

def make_color_wheel_image(size):
    hue = np.fromfunction(lambda i, j: (np.arctan2(i-size/2, size/2-j) + np.pi)*(180/np.pi)/2,
                          (size, size), dtype=np.float)

    # 255 if farthest from center. 0 if at center
    saturation = np.fromfunction(lambda i, j: (np.linalg.norm(np.array([i, j]) - np.array(size/2), axis=0) / (grid_size/2)) * 255, (size, size), dtype=np.float)

    value = np.ones((size, size)) * 150
    hsl = np.dstack((hue, saturation, value))
    color_map = cv2.cvtColor(np.array(hsl, dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return color_map

# get rid of white values by bumping values up by 1
norms = np.linalg.norm(pixel_values, axis=1, ord=2)[:, np.newaxis]
pixel_values = pixel_values[np.squeeze(norms > DARK_THRESH), :]

norms = np.linalg.norm(pixel_values, axis=1, ord=2)[:, np.newaxis]
pixel_values = pixel_values / norms

rgb = np.array([[-1/2, -np.sqrt(3)/2], [-1/2, np.sqrt(3)/2], [1, 0]]).astype(np.float64)
grid = []
for pix in pixel_values:
    grid.append(pix.dot(rgb))

grid_size = 1000
center = np.array((grid_size/2, grid_size/2))

# (-1, 0) -> (0, grid_size / 2)
# (1, 0) -> (grid_size, grid_size / 2)
# (0, -1) -> (grid_size / 2, grid_size)
grid = (np.array(grid) * np.array([1, -1]) + 1) / 2
grid = grid * grid_size

wheel = make_color_wheel_image(grid_size)

for coord in grid:
    x = int(coord[0])
    y = int(coord[1])
    if x < grid_size and x > 0 and y < grid_size and y > 0:
        wheel[y][x] = np.ones(3) * 255

wheel[grid_size - 1][0] = np.ones(3) * 255

im = Image.fromarray(wheel)
im.save(args.out)
