# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np
import pydicom
from pydicom.multival import MultiValue
from PIL import Image


def load_dcm_as_image(ds_file):
    ds = pydicom.dcmread(ds_file)
    # Apply Rescale Slope and Rescale Intercept if they exist
    pixel_array = ds.pixel_array.astype(float)
    rescale_slope = ds.get("RescaleSlope", 1)
    rescale_intercept = ds.get("RescaleIntercept", 0)
    if rescale_slope != 1 or rescale_intercept != 0:
        pixel_array = pixel_array * rescale_slope + rescale_intercept

    # Check for windowing information in the DICOM metadata
    window_center = ds.get("WindowCenter", None)
    window_width = ds.get("WindowWidth", None)

    if window_center is None or window_width is None:
        # If no windowing info, use the full dynamic range
        window_min = pixel_array.min()
        window_max = pixel_array.max()
    else:
        # Handle possible multi-valued tags by taking the first value
        if isinstance(window_center, MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, MultiValue):
            window_width = window_width[0]

        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

    # Apply windowing
    pixel_array = np.clip(pixel_array, window_min, window_max)

    # Normalize to 0-255
    if window_max > window_min:
        pixel_array = ((pixel_array - window_min) / (window_max - window_min)) * 255.0
    else:  # Handle case where all pixels are the same
        pixel_array.fill(128)

    # Handle Photometric Interpretation
    photometric_interpretation = ds.get("PhotometricInterpretation", "MONOCHROME2")
    if photometric_interpretation == "MONOCHROME1":
        pixel_array = 255.0 - pixel_array

    # Convert to 8-bit unsigned integer
    image_8bit = pixel_array.astype(np.uint8)

    return Image.fromarray(image_8bit)


if __name__ == '__main__':
    dcm_file = '../../example/images/image-00220.dcm'
    img = load_dcm_as_image(dcm_file)
    print(img)
    print(img.size)
    print(img.mode)
    img = np.array(img)
    print(img.shape)
