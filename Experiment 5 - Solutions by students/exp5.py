# EXPERIMENT 5
# Feature Extraction using SIFT and HOG
#
# Name: PALLAV PANKAJ
# Roll No: 40

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.feature import hog
from skimage import exposure


# ------------------------------------------------------------
# 1. Create results folder
# ------------------------------------------------------------

os.makedirs("results", exist_ok=True)

# Name: PALLAV PANKAJ
# Roll No: 40
# ------------------------------------------------------------
# 2. Load single image
# ------------------------------------------------------------

IMAGE_PATH = "image.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: image.jpg not found!")
    print("Put image.jpg in the same folder as exp5.py")
    exit()


# ------------------------------------------------------------
# 3. Convert image
# ------------------------------------------------------------

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


print("=" * 60)
print("SIFT AND HOG FEATURE EXTRACTION")
print("=" * 60)

print("\nImage size:", image.shape)


# ============================================================
# PART A - ORIGINAL IMAGE
# ============================================================

plt.figure(figsize=(7, 5))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.tight_layout()
plt.savefig("results/original_image.png", dpi=300)
plt.show()

# Name: PALLAV PANKAJ
# Roll No: 40
# ============================================================
# PART B - SIFT
# ============================================================

print("\n" + "=" * 60)
print("SIFT FEATURE EXTRACTION")
print("=" * 60)


# Create SIFT detector
sift = cv2.SIFT_create()


# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)


print("\nNumber of SIFT keypoints:", len(keypoints))

if descriptors is not None:
    print("SIFT descriptor shape:", descriptors.shape)


# Draw keypoints
sift_result = cv2.drawKeypoints(
    image_rgb,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# Display SIFT result
plt.figure(figsize=(10, 7))
plt.imshow(sift_result)
plt.title("SIFT Keypoints")
plt.axis("off")
plt.tight_layout()
plt.savefig("results/sift_keypoints.png", dpi=300)
plt.show()

# Name: PALLAV PANKAJ
# Roll No: 40
# ============================================================
# PART C - HOG
# ============================================================

print("\n" + "=" * 60)
print("HOG FEATURE EXTRACTION")
print("=" * 60)


# Resize image
hog_input = cv2.resize(gray, (256, 256))


# Extract HOG features
hog_features, hog_image = hog(
    hog_input,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm="L2-Hys"
)


# Rescale HOG visualization
hog_image = exposure.rescale_intensity(
    hog_image,
    in_range=(0, 10)
)


print("\nHOG feature vector length:",
      len(hog_features))


# Display HOG
plt.figure(figsize=(10, 7))
plt.imshow(hog_image, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")
plt.tight_layout()
plt.savefig("results/hog_visualization.png", dpi=300)
plt.show()

# Name: PALLAV PANKAJ
# Roll No: 40
# ============================================================
# PART D - SIFT AND HOG SIDE BY SIDE
# ============================================================

plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
plt.imshow(sift_result)
plt.title("SIFT Keypoints")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap="gray")
plt.title("HOG Features")
plt.axis("off")


plt.tight_layout()
plt.savefig("results/sift_vs_hog.png", dpi=300)
plt.show()


# ============================================================
# PART E - RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print("Number of SIFT Keypoints :", len(keypoints))
print("SIFT Descriptor Shape    :", descriptors.shape)
print("HOG Feature Vector Length:", len(hog_features))

print("\nSIFT detects distinctive keypoints.")
print("HOG represents edge and gradient information.")

print("\nOutput files saved in results folder.")

print("\nExperiment completed successfully.")

# Name: PALLAV PANKAJ
# Roll No: 40

# QUESTION AND ANSWER
# 
#  1. What is feature extraction, and why is it important in computer vision?

# Feature extraction is the process of converting raw image data into meaningful numerical features or descriptors that represent important characteristics of an image.

# It is important because it:

# Reduces complex image data into useful information.
# Helps in image classification and object recognition.
# Enables image matching and tracking.
# Makes images easier for machine-learning algorithms to analyze.
# Helps handle changes in scale, rotation, illumination, and viewpoint.
# 2. Explain the working principle of SIFT.

# SIFT (Scale-Invariant Feature Transform) is a feature extraction technique used to detect distinctive points in an image and generate descriptors for them.

# The main steps are:

# Scale-space construction – The image is repeatedly blurred using Gaussian filters at different scales.
# Keypoint detection – Difference of Gaussian (DoG) is used to identify potential keypoints.
# Keypoint localization – Unstable and low-contrast keypoints are removed.
# Orientation assignment – A dominant gradient orientation is assigned to each keypoint.
# Descriptor generation – A numerical descriptor is created around each keypoint based on local gradient information.
# Matching – Descriptors from two images can be compared to find corresponding features.

# SIFT is particularly useful because its features are robust to changes in scale and rotation.

# 3. What are keypoints and feature descriptors in image analysis?

# Keypoints are distinctive locations in an image, such as corners, edges, or textured regions, that can be reliably detected.

# Examples:

# Corners of objects
# Strong texture regions
# Distinctive patterns

# A feature descriptor is a numerical representation of the local image region surrounding a keypoint.

# For example, SIFT detects keypoints and generates a descriptor for each keypoint. These descriptors can then be compared between images for matching.

# In short:

# Keypoint = where an important feature is located
# Descriptor = numerical information describing that feature

# 4. Explain HOG and its significance.

# HOG (Histogram of Oriented Gradients) is a feature descriptor that represents an image based on the distribution of gradient directions.

# Working steps:

# Convert the image to grayscale.
# Calculate the horizontal and vertical image gradients.
# Calculate gradient magnitude and orientation.
# Divide the image into small cells.
# Create a histogram of gradient orientations for each cell.
# Group cells into blocks and normalize the histograms.
# Combine the normalized histograms to form the HOG feature vector.

# HOG is useful because it effectively represents the shape and edges of objects.

# It is commonly used for tasks such as object detection and recognition. The lab sheet specifically asks you to analyze HOG using parameters such as cell size, block size, and orientation bins.

# 5. Compare SIFT and HOG based on robustness, computational complexity, and applications.
# Feature	SIFT	HOG
# Full form	Scale-Invariant Feature Transform	Histogram of Oriented Gradients
# Representation	Local keypoints and descriptors	Global/local gradient histograms
# Scale invariance	Excellent	Limited
# Rotation invariance	Excellent	Limited
# Illumination robustness	Good	Good
# Computational complexity	Relatively high	Generally lower
# Best suited for	Image matching, recognition, tracking	Object detection and shape recognition
# Output	Keypoints + descriptors	Feature vector
# Example	Matching two photographs	Detecting pedestrians

# The lab sheet specifically requires comparison based on scale invariance, rotation invariance, and computational efficiency.

# 6. Why is SIFT considered invariant to scale and rotation?

# SIFT is scale invariant because it searches for keypoints across multiple image scales using a scale-space representation. Therefore, the same feature can be detected even if the image is enlarged or reduced.

# SIFT is rotation invariant because it assigns a dominant orientation to each keypoint. The descriptor is then constructed relative to this orientation.

# Therefore, rotating an image does not significantly change the resulting SIFT descriptor.

# 7. Mention three real-world applications where HOG descriptors are commonly used.

# Three applications are:

# Pedestrian detection – detecting people in surveillance and autonomous-driving systems.
# Vehicle detection – identifying vehicles based on their shape and edges.
# Object recognition – detecting objects based on their structural and gradient patterns.
# 8. Why is feature extraction performed before image classification or object detection?

# Raw images contain a very large amount of pixel information. Processing all pixels directly can be inefficient.

# Feature extraction:

# Reduces the dimensionality of the input.
# Removes less useful information.
# Captures important patterns such as edges, corners, textures, and shapes.
# Provides meaningful numerical representations.
# Makes classification and detection more efficient.

# Thus, extracted features can be supplied to a machine-learning algorithm for classification or detection.

# 9. What are the advantages and limitations of handcrafted feature descriptors compared to deep learning-based feature extraction?
# Handcrafted Features	Deep Learning Features
# Designed manually using algorithms	Learned automatically from data
# Usually require less training data	Usually require large datasets
# Lower computational requirements	Often require high computational resources
# Easier to understand and interpret	More difficult to interpret
# SIFT and HOG are examples	CNN-based features are examples
# Can work well for specific tasks	Often performs better on complex tasks

# Advantages of handcrafted descriptors:

# Computationally efficient in many applications.
# Do not require extensive training datasets.
# Interpretable.
# Useful for traditional computer-vision problems.

# Limitations:

# Designed using manually selected characteristics.
# May not perform well on highly complex images.
# Less adaptable than learned features.
# Often have lower performance than modern deep-learning methods on large-scale recognition tasks.
# 10. How do feature extraction techniques contribute to image matching, face recognition, and object detection?

# Image matching:
# SIFT extracts distinctive keypoints and descriptors from two images. Similar descriptors can be matched to determine whether the images contain the same object or scene.

# Face recognition:
# Feature extraction can identify important facial characteristics and convert them into numerical representations that can be compared with stored face features.

# Object detection:
# HOG can represent the shape and edge structure of objects. These features can then be provided to a classifier to detect objects such as pedestrians or vehicles.

# Therefore, feature extraction acts as an important intermediate step between raw image data and computer-vision tasks such as matching, recognition, and detection.
#  The experiment specifically asks students to apply SIFT/HOG features for basic image matching or object recognition using similar images. 