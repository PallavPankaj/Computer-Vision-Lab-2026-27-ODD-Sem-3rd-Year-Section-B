import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found!")
    exit()

print("Image loaded successfully!")

# Name: Pallav Pankaj
# Roll No.: 40


# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Name: Pallav Pankaj
# Roll No.: 40


# Contrast Stretching
min_pixel = np.min(gray_image)
max_pixel = np.max(gray_image)

contrast_stretched = ((gray_image - min_pixel) /
                   (max_pixel - min_pixel) * 255).astype(np.uint8)

# Name: Pallav Pankaj
# Roll No.: 40


# Histogram Equalization
equalized_image = cv2.equalizeHist(gray_image)

# Name: Pallav Pankaj
# Roll No.: 40


# CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(gray_image)

# Name: Pallav Pankaj
# Roll No.: 40


# Display Images
plt.figure(figsize=(5, 4))

plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretched")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()

# Name: Pallav Pankaj
# Roll No.: 40


# Compare Histograms
plt.figure(figsize=(5, 4))

plt.subplot(2, 2, 1)
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 2)
plt.hist(contrast_stretched.ravel(), 256, [0, 256])
plt.title("Contrast Stretched Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 3)
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 4)
plt.hist(clahe_image.ravel(), 256, [0, 256])
plt.title("CLAHE Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()

# Name: Pallav Pankaj
# Roll No.: 40


# Save Enhanced Images
cv2.imwrite("contrast_stretched.jpg", contrast_stretched)
cv2.imwrite("histogram_equalized.jpg", equalized_image)
cv2.imwrite("clahe_image.jpg", clahe_image)

print("Enhanced images saved successfully!")

# Name: Pallav Pankaj
# Roll No.: 40

plt.show()

#Observations

#   | Technique              | What to observe                                                          |
#| ---------------------- | ------------------------------------------------------------------------ |
#| Original               | Baseline image with its original contrast                                |
#| Contrast Stretching    | Wider intensity range and improved contrast                              |
#| Histogram Equalization | Redistribution of intensity values and enhanced overall contrast         |
#| CLAHE                  | Local contrast enhancement, particularly useful when illumination varies |


# QUESTIONS AND ANSWERS:

# """ 1. What is image contrast, and why is it important in image processing?

# Answer:
# Image contrast is the difference in intensity between the dark and bright regions of an image. High-contrast images have clearly distinguishable dark and bright areas, while low-contrast images appear dull or unclear.
# Contrast is important because it improves the visibility of objects, edges, and details. Better contrast also helps computer vision systems perform tasks such as feature extraction, segmentation, and object recognition more effectively.



# 2. Explain the concept of an image histogram. What information does it provide?

# Answer:
# An image histogram is a graphical representation of the distribution of pixel intensity values in an image.
# For a grayscale image, intensity values normally range from 0 to 255, where 0 represents black and 255 represents white.

# A histogram provides information about:

# The distribution of brightness levels.
# Whether an image is dark or bright.
# The contrast of the image.
# The concentration of pixels within particular intensity ranges.

# It is useful for analyzing and improving image quality.



# 3. Differentiate between Histogram Stretching and Histogram Equalization.

# Answer:

# Histogram Stretching	Histogram Equalization
# Expands the existing intensity range.	Redistributes pixel intensities across the available range.
# Improves contrast by mapping minimum and maximum values.	Improves contrast using the cumulative distribution of pixel intensities.
# Usually preserves the general shape of the histogram.	Can significantly change the histogram distribution.
# Simple and useful for low-contrast images.	Useful when pixel intensities are concentrated in a limited range.

# In this experiment, histogram stretching was implemented manually, while histogram equalization was performed using OpenCV.

# 4. What is Contrast Limited Adaptive Histogram Equalization (CLAHE)? How does it differ from standard Histogram Equalization?

# Answer:
# CLAHE stands for Contrast Limited Adaptive Histogram Equalization. It enhances contrast by dividing an image into small regions called tiles and performing histogram equalization separately on these regions.

# The main difference is:
# Standard Histogram Equalization: Works globally on the entire image.
# CLAHE: Works locally on different regions of the image.

# CLAHE also limits excessive contrast enhancement using a clip limit, which helps prevent noise from being overly amplified.

# Therefore, CLAHE is particularly useful for images having varying illumination or local contrast problems.



# 5. Why is histogram equalization commonly applied before feature extraction and image segmentation?

# Answer:
# Histogram equalization improves the distribution of pixel intensities and increases image contrast. This makes important features such as edges, boundaries, and objects more visible.
# As a result, feature extraction algorithms can identify useful features more effectively, while segmentation algorithms can better distinguish between different regions of an image.
# Therefore, histogram equalization can improve the quality of input data before subsequent computer vision operations.



# 6. Mention three real-world applications where histogram equalization is widely used.

# Answer:
# Three real-world applications are:

# Medical imaging — improving the visibility of structures and details in medical images.
# Satellite and remote-sensing images — enhancing features in satellite imagery.
# Low-light photography — improving visibility and contrast in poorly illuminated photographs.

# These applications are consistent with the types of real-world images suggested in the experiment.




# 7. What are the limitations of global histogram equalization?

# Answer:
# The main limitations of global histogram equalization are:

# It considers the entire image rather than local regions.
# It may over-enhance some areas.
# It can amplify noise.
# It may produce unnatural-looking images.
# It may not perform well when different parts of an image have different illumination conditions.

# CLAHE can help address some of these problems by performing enhancement locally and limiting excessive contrast enhancement.




# 8. How does contrast enhancement improve the performance of object detection and recognition systems?

# Answer:
# Contrast enhancement makes important image features such as edges, boundaries, textures, and objects more visible.

# This can help object detection and recognition systems by:

# Improving feature visibility.
# Making object boundaries clearer.
# Improving the distinction between objects and backgrounds.
# Providing better input for feature extraction algorithms.

# Therefore, contrast enhancement can make subsequent computer vision tasks more effective, especially when the original image has poor contrast.




# 9. Compare histogram-based enhancement techniques with brightness adjustment methods.

# Answer:

# Histogram-Based Enhancement	Brightness Adjustment
# Changes the distribution of pixel intensities.	Mainly shifts pixel intensities toward brighter or darker values.
# Can improve image contrast.	Primarily changes overall brightness.
# Includes histogram stretching, histogram equalization, and CLAHE.	Usually involves adding or subtracting an intensity value.
# Can reveal details that are difficult to see in low-contrast images.	May not improve contrast significantly.
# More suitable when intensity distribution needs improvement.	Suitable when the main problem is excessive darkness or brightness.

# Thus, histogram-based techniques generally provide more control over contrast than simple brightness adjustment.





# 10. Why is CLAHE preferred for medical imaging and low-light image enhancement?

# Answer:
# CLAHE is preferred because it performs local contrast enhancement instead of applying the same enhancement to the entire image.

# This is useful for medical and low-light images because different regions may have different illumination levels. CLAHE can enhance details in darker regions while limiting excessive contrast enhancement.

# It also uses a clip limit, which helps reduce excessive amplification of noise.

# Therefore, CLAHE is particularly useful when an image contains varying illumination and important local details need to be enhanced.  
# """