from PIL import Image
import numpy as np
from random import random
from tqdm import tqdm

# Global parameters
matrix_size = 513  # must be 2**n + 1
random_division = 1.3
max_color = 255  # Equivalent to 'Max Saturation' in the original code

def random_range(low, high):
    return (random() * (high - low)) + low

def generate_image(image_name):
    mat = np.zeros((matrix_size, matrix_size))

    # Diamond-square algorithm
    space = matrix_size - 1
    halfspace = 0
    random_factor = 100  # Initial random variation

    # Initialize corners
    mat[0][0] = random_range(-random_factor, random_factor)
    mat[0][matrix_size - 1] = random_range(-random_factor, random_factor)
    mat[matrix_size - 1][0] = random_range(-random_factor, random_factor)
    mat[matrix_size - 1][matrix_size - 1] = random_range(-random_factor, random_factor)

    while space > 1:
        halfspace = space // 2

        # Diamond step
        for x in range(halfspace, matrix_size, space):
            for y in range(halfspace, matrix_size, space):
                avg = (
                    mat[x - halfspace][y - halfspace]
                    + mat[x - halfspace][y + halfspace]
                    + mat[x + halfspace][y - halfspace]
                    + mat[x + halfspace][y + halfspace]
                ) / 4
                mat[x][y] = avg + random_range(-random_factor, random_factor)

        # Square step
        offset = 0
        for x in range(0, matrix_size, halfspace):
            offset = halfspace if offset == 0 else 0
            for y in range(offset, matrix_size, space):
                s, n = 0, 0
                if x >= halfspace:
                    s += mat[x - halfspace][y]
                    n += 1
                if x + halfspace < matrix_size:
                    s += mat[x + halfspace][y]
                    n += 1
                if y >= halfspace:
                    s += mat[x][y - halfspace]
                    n += 1
                if y + halfspace < matrix_size:
                    s += mat[x][y + halfspace]
                    n += 1
                avg = s / n
                mat[x][y] = avg + random_range(-random_factor, random_factor)

        random_factor /= random_division
        space = halfspace

    # Rescale altitudes and set pixels
    min_altitude = min(min(row) for row in mat)
    max_altitude = max(max(row) for row in mat)

    for x in range(matrix_size):
        for y in range(matrix_size):
            mat[x][y] = int(((mat[x][y] - min_altitude) / (max_altitude - min_altitude)) * max_color)
    
    mat = mat.astype(np.uint8)
    im = Image.fromarray(mat, mode="L")
    im.save(f"images/{image_name}.png", compress_level=0)


for i in tqdm(range(1000)):
    generate_image(f"image_{i}")
