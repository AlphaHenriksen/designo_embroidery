from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import sys


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array(list(int(h[i : i + 2], 16) for i in (0, 2, 4)))


def ColorDistance(rgb1, rgb2):
    """d = {} distance between two colors(3)"""
    rm = 0.5 * (rgb1[0] + rgb2[0])
    d = np.sqrt(abs(sum((2 + rm, 4, 3 - rm) * (rgb1 - rgb2) ** 2)))
    return d


def ColorDistanceMat(rgb1, rgb2):
    """d = {} distance between two colors(3)
    the second input is now an array of colors instead of just one color"""
    rm = 0.5 * (rgb1[0] + rgb2[:, 0])
    d = np.sqrt(
        abs(
            np.sum(
                np.array([2 + rm, np.repeat(4, np.shape(rgb2)[0]), 3 - rm]).T
                * np.square((rgb1 - rgb2)),
                1,
            )
        )
    )
    return d


# Load the file full of DMC colors
with open("embroidery_helper/DMC_colors.json", "r") as f:
    dmcs_hex = json.load(f)

dmcs_rgb = np.array(
    [hex2rgb(hex) for hex in dmcs_hex.keys()]
)  # Convert all of them to rgb values
dmcs_rgb = dmcs_rgb.flatten()
print(dmcs_rgb)


# Load the image
filepath = "embroidery_helper/stephanie.jpg"
img = Image.open(filepath)
rows, cols, channels = np.shape(np.array(img))
print(rows, cols, channels)
rows = rows // 32
cols = cols // 32
print(rows, cols, channels)
max_size = (rows, cols)
img.thumbnail(max_size)
print(img.size)
# plt.imshow(img)
# plt.show()
img = img.convert("P", palette=Image.ADAPTIVE, colors=28)
plt.imshow(img)
plt.show()

# PALETTE = [
#     0,
#     0,
#     0,  # black,  00
#     0,
#     255,
#     0,  # green,  01
#     255,
#     0,
#     0,  # red,    10
#     255,
#     255,
#     0,  # yellow, 11
# ]


# # a palette image to use for quant
# print(list(dmcs_rgb[: 256 * 3]))
# pimage = Image.new("P", (1, 1), 0)
# pimage.putpalette(dmcs_rgb)

# # open the source image
# image = Image.open(filepath)
# # plt.imshow(image)
# # plt.show()
# image = image.convert("RGB")
# # image.show()

# # quantize it using our palette image
# imagep = image.quantize(palette=pimage)
# # plt.imshow(imagep)
# plt.show()
# imagep.show()

# save
# imagep.save('/tmp/cga.png')


# pixels = np.array(img.convert("RGBA").getdata())  # Extract every pixel
# new_pixels = np.zeros_like(pixels[:, :-1])


# for i, pixel in enumerate(tqdm(pixels[:, :-1])):  # Remove the alpha-value
#     col_idx = np.argmin(ColorDistanceMat(pixel, dmcs_rgb))
#     new_pixels[i] = dmcs_rgb[col_idx, :]

# file = open("embroidery_helper/arr", "wb")
# np.save(file, new_pixels)
# # array = np.load(file)
# print(new_pixels)

# new_pixels = new_pixels.astype(np.uint8)
# new_img_arr = np.reshape(new_pixels, (img.size[0], img.size[1], channels))
# new_img = Image.fromarray(new_img_arr)
# new_img.save("embroidery_helper/new.jpg")
# new_img.show()
# file.close()
