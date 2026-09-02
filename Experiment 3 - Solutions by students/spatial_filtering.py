import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Name    : PALLAV PANKAJ
# Roll No.: 40

INPUT_IMAGE = "input.jpg"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_pixels = int(amount * image.size)

    salt_coords = (
        np.random.randint(0, image.shape[0], num_pixels),
        np.random.randint(0, image.shape[1], num_pixels)
    )
    noisy[salt_coords] = 255

    pepper_coords = (
        np.random.randint(0, image.shape[0], num_pixels),
        np.random.randint(0, image.shape[1], num_pixels)
    )
    noisy[pepper_coords] = 0
    return noisy

def save_image(name, image):
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), image)

# Name    : PALLAV PANKAJ
# Roll No.: 40

img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(
        "input.jpg not found. Put an image named 'input.jpg' in this folder."
    )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noisy = add_salt_pepper_noise(gray)

# Low-pass filters
start = time.perf_counter()
average = cv2.blur(noisy, (5, 5))
average_time = time.perf_counter() - start

# Name    : PALLAV PANKAJ
# Roll No.: 40

start = time.perf_counter()
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
gaussian_time = time.perf_counter() - start

start = time.perf_counter()
median = cv2.medianBlur(noisy, 5)
median_time = time.perf_counter() - start

# High-pass filters
start = time.perf_counter()
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
laplacian_time = time.perf_counter() - start

start = time.perf_counter()
sobel_x_float = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = np.uint8(np.absolute(sobel_x_float))
sobel_x_time = time.perf_counter() - start

# Name    : PALLAV PANKAJ
# Roll No.: 40

start = time.perf_counter()
sobel_y_float = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = np.uint8(np.absolute(sobel_y_float))
sobel_y_time = time.perf_counter() - start

sobel_combined = cv2.magnitude(
    sobel_x_float.astype(np.float32),
    sobel_y_float.astype(np.float32)
)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

# Name    : PALLAV PANKAJ
# Roll No.: 40

# Save outputs
save_image("original_gray.jpg", gray)
save_image("noisy.jpg", noisy)
save_image("average_filter.jpg", average)
save_image("gaussian_filter.jpg", gaussian)
save_image("median_filter.jpg", median)
save_image("laplacian_filter.jpg", laplacian)
save_image("sobel_x.jpg", sobel_x)
save_image("sobel_y.jpg", sobel_y)
save_image("sobel_combined.jpg", sobel_combined)

# Comparison figure
plt.figure(figsize=(16, 10))
images = [
    (gray, "Original"),
    (noisy, "Salt-and-Pepper Noise"),
    (average, "Average Filter"),
    (gaussian, "Gaussian Filter"),
    (median, "Median Filter"),
    (laplacian, "Laplacian"),
    (sobel_x, "Sobel X"),
    (sobel_y, "Sobel Y"),
]
for i, (image, title) in enumerate(images, 1):
    plt.subplot(2, 4, i)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison.png"), dpi=200)
plt.close()

# Name    : PALLAV PANKAJ
# Roll No.: 40

print("=" * 55)
print("SPATIAL FILTERING EXPERIMENT - COMPLETED")
print("=" * 55)
print(f"Average Filter   : {average_time:.6f} seconds")
print(f"Gaussian Filter  : {gaussian_time:.6f} seconds")
print(f"Median Filter    : {median_time:.6f} seconds")
print(f"Laplacian Filter : {laplacian_time:.6f} seconds")
print(f"Sobel X          : {sobel_x_time:.6f} seconds")
print(f"Sobel Y          : {sobel_y_time:.6f} seconds")
print(f"Results saved in : {OUTPUT_DIR}/")
print("=" * 55)

# Name    : PALLAV PANKAJ
# Roll No.: 40

# Questions and Answers


# Q1. What is spatial filtering? How is it used in digital image processing?

# Answer:
# Spatial filtering is an image-processing technique in which the value of a pixel is modified based on the values of neighboring pixels.
# A small matrix called a kernel or filter mask is moved across the image. Mathematical operations are performed between the kernel and the corresponding image pixels to produce a new pixel value.

# Spatial filtering is used for:
# Noise reduction
# Image smoothing
# Edge detection
# Sharpening
# Feature enhancement
# Image preprocessing

# It is an important operation in digital image processing because it directly modifies pixel values based on their local neighborhood.


# Q2. Differentiate between Low-Pass Filters and High-Pass Filters with suitable examples.

# Low-Pass Filter	                     High-Pass Filter
# Smooths an image	                     Enhances edges/details
# Reduces high-frequency components	     Emphasizes high-frequency components
# Used for noise reduction	             Used for edge detection
# Can cause image blurring	             Can make edges more prominent
# Examples: Average, Gaussian, Median	 Examples: Laplacian, Sobel

# Low-pass filters primarily reduce noise and smooth images, whereas high-pass filters emphasize edges, fine details, and abrupt intensity changes.



# Q3. Compare Average Filter, Gaussian Filter, and Median Filter.

# Feature             Average	                         Gaussian	                         Median
# Principle	          Calculates neighborhood mean	     Uses Gaussian-weighted average	     Uses neighborhood median
# Type	              Low-pass	                         Low-pass	                         Non-linear low-pass
# Noise reduction	  Good	                             Very good	                         Excellent for salt-and-pepper
# Edge preservation	  Poor	                             Better	                             Excellent
# Main use	          Basic smoothing	                 Natural image smoothing	         Impulse-noise removal

# The experiment specifically compares Average and Gaussian smoothing and uses Median Filtering for impulse noise removal.


# Q4. Why is the Median Filter particularly effective for removing salt-and-pepper noise?

# Answer:

# Salt-and-pepper noise appears as isolated black and white pixels.
# The Median Filter replaces a pixel with the median value of its neighborhood rather than the average.
# Extreme values such as 0 and 255 have less influence on the median. Therefore, noisy pixels can be removed while preserving important edges.
# For this reason, the Median Filter is particularly effective for salt-and-pepper noise.



# Q5. Explain the role of convolution kernels in spatial filtering.

# Answer:
# A convolution kernel is a small matrix containing numerical coefficients.
# During spatial filtering, the kernel moves across the image. At every position:
# The kernel overlaps a neighborhood of pixels.
# Corresponding pixel and kernel values are multiplied.
# The products are added.
# The resulting value becomes the new pixel value.
# Different kernels perform different operations.

# For example:

# Average kernel:

# 1/9  1/9  1/9
# 1/9  1/9  1/9
# 1/9  1/9  1/9

# Sobel X kernel:

# -1   0   1
# -2   0   2
# -1   0   1

# Thus, kernels determine the behavior of a spatial filter.



# Q6. What is the purpose of the Sobel and Laplacian operators in edge detection?

# Answer:

# The Sobel operator detects image gradients in particular directions.

# Sobel X detects changes in the horizontal direction.
# Sobel Y detects changes in the vertical direction.

# The Laplacian operator uses second-order derivatives and detects rapid intensity changes in multiple directions.

# Therefore:

# Sobel → directional gradient detection
# Laplacian → general edge and fine-detail detection

# The experiment explicitly uses both operators for edge enhancement and gradient detection.



# Q7. Why are filtering operations considered an essential preprocessing step in computer vision?

# Answer:

# Real-world images often contain noise, unwanted details, illumination variations, or other distortions.
# Filtering can improve the input before further computer-vision operations.

# For example:

# Raw Image
#     ↓
# Noise Reduction
#     ↓
# Filtering
#     ↓
# Edge Detection
#     ↓
# Feature Extraction
#     ↓
# Object Recognition

# Filtering therefore improves image quality and can make subsequent computer-vision algorithms more reliable.

# Applications include:
# Medical imaging
# Surveillance
# Remote sensing
# Autonomous systems

# These applications are also highlighted in the experiment document.



# Q8. Discuss the trade-off between image smoothing and edge preservation during filtering.

# Answer:

# Image smoothing reduces noise but can also remove useful image details.
# For example, using a very large smoothing kernel can significantly reduce noise, but it may also blur edges and fine structures.
# On the other hand, using a smaller kernel preserves more details but may not remove enough noise.

# Therefore, a balance must be maintained:

# More Smoothing
#       ↓
# Less Noise
#       +
# More Blurring

# while:

# Less Smoothing
#       ↓
# More Details
#       +
# More Noise

# The best filter and kernel size depend on the application and the type of noise present.


# Q9. Mention four real-world applications where spatial filtering techniques are widely used.

# Answer:

# Four major applications are:
# Medical Imaging – noise reduction and enhancement of structures in medical images.
# Surveillance Systems – image enhancement and edge detection.
# Remote Sensing – enhancement and analysis of satellite/aerial imagery.
# Autonomous Systems – preprocessing camera images for object and feature detection.

# These application areas are specifically mentioned in the experiment description.


# Q10. Compare spatial domain filtering with frequency domain filtering.

# Spatial Domain Filtering	Frequency Domain Filtering
# Works directly on image pixels	Works on frequency components
# Uses kernels/masks	Uses frequency transformations
# Relatively intuitive and straightforward	Requires transformation to frequency domain
# Suitable for local operations	Useful for analyzing/removing frequency components
# Examples: Gaussian, Median, Sobel	Fourier-transform-based filtering
# Spatial Domain

# In spatial filtering, operations are directly performed on neighboring pixels using kernels.

# Frequency Domain

# In frequency-domain filtering, the image is transformed into a frequency representation, filtering is performed there, and the result is transformed back.

# Spatial filtering is often convenient for local operations such as smoothing and edge detection, while frequency-domain methods are useful when filtering specific frequency components is advantageous.