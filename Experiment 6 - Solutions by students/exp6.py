import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# EXPERIMENT NO. 6
# IMPLEMENTATION AND COMPARATIVE ANALYSIS OF
# IMAGE SEGMENTATION TECHNIQUES
#
# Name: PALLAV PANKAJ
# Roll No.: 40

# ------------------------------------------------------------
# 1. Load Image
# ------------------------------------------------------------

IMAGE_PATH = "input.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Image not found!")
    print("Make sure 'input.jpg' is in the same folder as this program.")
    exit()

# Convert BGR to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Name: PALLAV PANKAJ
# Roll No.: 40
# ------------------------------------------------------------
# 2. Convert Image to Grayscale
# ------------------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# 3. Gaussian Blur - Noise Reduction
# ------------------------------------------------------------

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ============================================================
# 4. GLOBAL THRESHOLDING
# ============================================================

_, global_thresh = cv2.threshold(
    blur,
    127,
    255,
    cv2.THRESH_BINARY
)
# Name: PALLAV PANKAJ
# Roll No.: 40
# ============================================================
# 5. OTSU'S THRESHOLDING
# ============================================================

otsu_threshold, otsu_thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Otsu Optimal Threshold:", otsu_threshold)

# ============================================================
# 6. ADAPTIVE THRESHOLDING
# ============================================================

adaptive_thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)
# Name: PALLAV PANKAJ
# Roll No.: 40
# ============================================================
# 7. WATERSHED SEGMENTATION
# ============================================================

# Convert threshold image into binary image
_, binary = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Remove noise using morphological opening
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

# Determine sure background
sure_bg = cv2.dilate(
    opening,
    kernel,
    iterations=3
)

# Determine sure foreground
dist_transform = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)

_, sure_fg = cv2.threshold(
    dist_transform,
    0.5 * dist_transform.max(),
    255,
    0
)

sure_fg = np.uint8(sure_fg)

# Unknown region
unknown = cv2.subtract(
    sure_bg,
    sure_fg
)

# Marker labelling
num_markers, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1

markers[unknown == 255] = 0

# Apply Watershed
watershed_image = image.copy()

markers = cv2.watershed(
    watershed_image,
    markers
)

# Mark boundaries in red
watershed_image[markers == -1] = [0, 0, 255]

watershed_rgb = cv2.cvtColor(
    watershed_image,
    cv2.COLOR_BGR2RGB
)
# Name: PALLAV PANKAJ
# Roll No.: 40
# ============================================================
# 8. K-MEANS CLUSTERING
# ============================================================

# Reshape image pixels
pixels = image_rgb.reshape(
    (-1, 3)
)

pixels = np.float32(pixels)

# Number of clusters
K = 3

# Apply K-Means
kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(pixels)

centers = np.uint8(
    kmeans.cluster_centers_
)

# Replace pixels with cluster centers
segmented_pixels = centers[labels]

kmeans_image = segmented_pixels.reshape(
    image_rgb.shape
)

# ============================================================
# 9. DISPLAY ALL RESULTS
# ============================================================

plt.figure(figsize=(14, 10))

plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(global_thresh, cmap="gray")
plt.title("Global Threshold")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(otsu_thresh, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("Adaptive Threshold")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(watershed_rgb)
plt.title("Watershed")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(kmeans_image)
plt.title("K-Means")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.tight_layout()
plt.show()
# Name: PALLAV PANKAJ
# Roll No.: 40
# ============================================================
# 10. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

cv2.imwrite(
    "results/grayscale.jpg",
    gray
)

cv2.imwrite(
    "results/global_threshold.jpg",
    global_thresh
)

cv2.imwrite(
    "results/otsu_threshold.jpg",
    otsu_thresh
)

cv2.imwrite(
    "results/adaptive_threshold.jpg",
    adaptive_thresh
)

cv2.imwrite(
    "results/watershed.jpg",
    watershed_image
)

cv2.imwrite(
    "results/kmeans.jpg",
    cv2.cvtColor(
        kmeans_image,
        cv2.COLOR_RGB2BGR
    )
)

print("\nAll segmentation results have been saved in the 'results' folder.")
# Name: PALLAV PANKAJ
# Roll No.: 40

#QUESTIONS:

# 1. What is image segmentation? Why is it considered a fundamental step in computer vision?

# Image segmentation is the process of dividing an image into meaningful regions or objects based on properties such as intensity, color, texture, or boundaries.

# It is fundamental because it isolates the Region of Interest (ROI) from the background, making subsequent tasks such as object detection, recognition, medical diagnosis, and autonomous navigation easier and more accurate.

# 2. Differentiate between Image Segmentation and Image Classification.

# Image Segmentation	Image Classification
# Divides an image into regions or objects.	Assigns a class/label to an entire image or object.
# Produces pixel-level or region-level output.	Usually produces a single class label.
# Identifies where an object is.	Identifies what the object is.
# Example: separating a tumor from surrounding tissue.	Example: classifying an X-ray as normal or abnormal.

# 3. Explain Global Thresholding, Otsu's Thresholding, and Adaptive Thresholding.

# Global Thresholding

# A single threshold value is selected for the entire image.
# It works well when the foreground and background have significantly different intensities.

# Otsu's Thresholding
# Otsu's method automatically determines the optimal threshold from the image histogram. It selects the threshold that maximizes between-class variance between foreground and background.
# It is useful when the image has a reasonably bimodal histogram.

# Adaptive Thresholding
# Adaptive thresholding calculates a different threshold for different local regions of the image.
# It is useful when illumination is uneven or changes across the image.

# 4. What is the Watershed Algorithm? Why is it useful for separating overlapping objects?

# Watershed is a region-based segmentation algorithm that treats an image like a geographical surface containing valleys and ridges.
# The algorithm identifies different regions starting from markers and expands them until boundaries meet.

# It is particularly useful for separating touching or overlapping objects, such as multiple cells or coins that appear connected in an image.

# 5. How is K-Means Clustering applied to image segmentation?

# K-Means groups pixels into K clusters according to their feature similarity.

# Steps:

# Select the number of clusters \(K\).
# Represent each pixel using features such as RGB values.
# Randomly initialize \(K\) cluster centers.
# Assign each pixel to its nearest cluster center.
# Recalculate cluster centers.
# Repeat until the centers stabilize.
# Replace each pixel with its cluster-center value.

# Thus, pixels having similar colors are grouped into the same region. The lab specifically uses K-Means for color-based segmentation.

# 6. Compare threshold-based and clustering-based segmentation.

# Threshold-Based	Clustering-Based
# Uses intensity threshold values.	Groups pixels based on similarity.
# Usually simpler and faster.	Generally more computationally expensive.
# Works well with clear foreground/background contrast.	Can handle multiple regions/colors.
# Global/adaptive thresholds are commonly used.	K-Means is a common approach.
# Sensitive to illumination and threshold selection.	Sensitive to number of clusters and initialization.

# 7. What challenges are encountered while segmenting images with complex backgrounds or varying illumination?

# Major challenges include:

# Similar colors between foreground and background.
# Uneven illumination.
# Shadows and reflections.
# Image noise.
# Objects touching or overlapping.
# Complex textures.
# Poor contrast.
# Gradual rather than sharp object boundaries.

# Adaptive thresholding can help with varying illumination, while Watershed can help separate touching objects.

# 8. Mention five real-world applications of image segmentation.

# Five important applications are:

# Medical imaging — detecting and isolating tumors or organs.
# Autonomous vehicles — identifying roads, vehicles, pedestrians, and obstacles.
# Satellite imagery — separating buildings, vegetation, water bodies, and roads.
# Industrial inspection — detecting defects in manufactured products.
# Object detection and recognition — isolating objects from their backgrounds.

# These applications align with the lab sheet's stated uses, including medical diagnosis, autonomous navigation, and industrial inspection.

# 9. How does image segmentation improve object detection and image recognition systems?

# Segmentation isolates the important objects or regions from irrelevant background information.

# This can:

# Reduce background noise.
# Improve object boundaries.
# Provide more accurate object regions.
# Reduce unnecessary image information.
# Improve feature extraction.
# Help recognition algorithms focus on the Region of Interest.

# Therefore, segmentation can improve the accuracy and efficiency of later computer-vision tasks.

# 10. Compare traditional segmentation techniques with U-Net and Mask R-CNN.

# Traditional Techniques	Deep Learning Techniques
# Examples: Thresholding, Otsu, Watershed, K-Means.	Examples: U-Net, Mask R-CNN.
# Usually require manually selected parameters or rules.	Learn segmentation features from training data.
# Simple and relatively fast.	Require more computational resources and training.
# Work well for simple/controlled images.	Better suited to complex real-world images.
# Limited ability to understand high-level context.	Can learn complex shapes, textures, and contextual information.
# Usually do not require large labeled datasets.	Generally require labeled training data.

# U-Net is primarily designed for pixel-level semantic segmentation and is widely useful for medical images.

# Mask R-CNN performs instance segmentation, meaning it can identify and create a separate mask for each individual object.

# The traditional techniques in this experiment are intended to be compared based on object boundaries, accuracy, and computational efficiency.