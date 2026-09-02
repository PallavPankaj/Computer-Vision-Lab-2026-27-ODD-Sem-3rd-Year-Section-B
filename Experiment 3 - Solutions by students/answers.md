# Questions and Answers

## 1. What is spatial filtering?
Spatial filtering modifies pixel values using neighboring pixels. It is used for smoothing, noise reduction, sharpening, edge detection, and feature enhancement.

## 2. Low-Pass vs High-Pass Filters
Low-pass filters smooth images and reduce high-frequency noise. Examples: Average, Gaussian, Median.
High-pass filters emphasize edges and fine details. Examples: Laplacian and Sobel.

## 3. Average, Gaussian, and Median
Average uses the neighborhood mean and may blur edges. Gaussian uses weighted averaging and provides smoother results. Median replaces a pixel with the neighborhood median and is particularly effective against salt-and-pepper noise.

## 4. Why is Median effective for salt-and-pepper noise?
Salt-and-pepper noise contains extreme black and white pixels. The median is less affected by these extreme values, so noise is removed while edges are relatively well preserved.

## 5. Role of convolution kernels
A kernel is a small matrix moved across the image. Pixel neighborhoods are combined with kernel coefficients to calculate filtered pixel values.

## 6. Sobel and Laplacian
Sobel detects directional gradients (X and Y). Laplacian detects rapid intensity changes and highlights edges and fine details.

## 7. Why preprocessing?
Filtering reduces noise and unwanted variations before feature extraction, edge detection, segmentation, or recognition.

## 8. Smoothing vs edge preservation
More smoothing generally reduces more noise but can blur useful edges and details. Less smoothing preserves detail but may leave more noise.

## 9. Four applications
1. Medical imaging
2. Surveillance
3. Remote sensing
4. Autonomous systems

## 10. Spatial vs frequency domain
Spatial filtering operates directly on image pixels using local kernels. Frequency-domain filtering transforms an image into frequency components, modifies those components, and transforms the result back.
