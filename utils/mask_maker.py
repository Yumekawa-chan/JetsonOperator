from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import opening,closing, disk
from skimage.filters import threshold_otsu

def make_mask(base_image_path, input_image_path, id):
    base_image = Image.open(base_image_path).convert('L')
    input_image = Image.open(input_image_path).convert('L')

    blur_sigma = 2
    opening_size = 1
    closing_size = 1

    base_array = np.array(base_image)
    input_array = np.array(input_image)

    base_array_blurred = gaussian_filter(base_array, sigma=blur_sigma)
    input_array_blurred = gaussian_filter(input_array, sigma=blur_sigma)

    diff_array = np.abs(base_array_blurred - input_array_blurred)

    threshold = threshold_otsu(diff_array)
    threshold = 40

    mask_array = np.where(diff_array > threshold, 255, 0).astype(np.uint8)

    selem = disk(opening_size)
    mask_cleaned = opening(mask_array, selem)
    mask_cleaned = closing(mask_cleaned, disk(closing_size))

    mask_image = Image.fromarray(mask_cleaned)
    mask_image.save(f'./mask/mask_{id}.png')
