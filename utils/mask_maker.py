from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import opening, disk

def create_difference_mask(base_image_path, input_image_path, threshold, blur_sigma=2, opening_size=3):
    base_image = Image.open(base_image_path).convert('L')
    input_image = Image.open(input_image_path).convert('L')

    blur_sigma=2
    opening_size=3
    threshold=30

    base_array = np.array(base_image)
    input_array = np.array(input_image)

    base_array_blurred = gaussian_filter(base_array, sigma=blur_sigma)
    input_array_blurred = gaussian_filter(input_array, sigma=blur_sigma)

    diff_array = np.abs(base_array_blurred - input_array_blurred)

    mask_array = np.where(diff_array > threshold, 255, 0).astype(np.uint8)

    selem = disk(opening_size)
    mask_cleaned = opening(mask_array, selem)

    mask_image = Image.fromarray(mask_cleaned)
    
    return mask_image.save('difference_mask.png')



