# EXPERIMENT NO. 4
# FREQUENCY DOMAIN IMAGE FILTERING USING FOURIER TRANSFORM

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ------------------------------------------------------------
# 1. LOAD IMAGE
# ------------------------------------------------------------

INPUT_IMAGE = "input.jpg"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read image in grayscale
image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f"Image '{INPUT_IMAGE}' not found. "
        "Place the image in the same folder as this Python file."
    )

print("Image loaded successfully.")
print("Image Shape:", image.shape)

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 2. COMPUTE DISCRETE FOURIER TRANSFORM (DFT)
# ------------------------------------------------------------

# Convert image to floating-point format
image_float = np.float32(image)

# Compute 2D Fourier Transform
dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)

print("DFT computed successfully.")


# ------------------------------------------------------------
# 3. SHIFT ZERO-FREQUENCY COMPONENT TO CENTER
# ------------------------------------------------------------

# Move low-frequency components to the center
dft_shift = np.fft.fftshift(dft)

print("Zero-frequency component shifted to center.")

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 4. CALCULATE MAGNITUDE SPECTRUM
# ------------------------------------------------------------

# Calculate magnitude of the frequency components
magnitude_spectrum = cv2.magnitude(
    dft_shift[:, :, 0],
    dft_shift[:, :, 1]
)

# Use logarithmic scale for better visualization
magnitude_spectrum = np.log(magnitude_spectrum + 1)

# Normalize spectrum to 0-255
magnitude_spectrum = cv2.normalize(
    magnitude_spectrum,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

magnitude_spectrum = np.uint8(magnitude_spectrum)

print("Magnitude spectrum generated.")

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 5. CREATE LOW-PASS FILTER
# ------------------------------------------------------------

rows, cols = image.shape

# Center coordinates
crow = rows // 2
ccol = cols // 2

# Radius of Low-Pass Filter
LOW_PASS_RADIUS = 50

# Create a mask containing zeros
low_pass_mask = np.zeros((rows, cols, 2), np.float32)

# Create circular Low-Pass Filter
cv2.circle(
    low_pass_mask,
    (ccol, crow),
    LOW_PASS_RADIUS,
    (1, 1),
    -1
)

print("Low-Pass Filter created.")

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 6. APPLY LOW-PASS FILTER
# ------------------------------------------------------------

# Multiply DFT by Low-Pass mask
low_pass_dft = dft_shift * low_pass_mask

# Shift frequency components back
low_pass_dft_shift = np.fft.ifftshift(low_pass_dft)

# Apply Inverse DFT
low_pass_result = cv2.idft(low_pass_dft_shift)

# Calculate magnitude
low_pass_result = cv2.magnitude(
    low_pass_result[:, :, 0],
    low_pass_result[:, :, 1]
)

# Normalize image
low_pass_result = cv2.normalize(
    low_pass_result,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

low_pass_result = np.uint8(low_pass_result)

print("Low-Pass filtering completed.")

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 7. CREATE HIGH-PASS FILTER
# ------------------------------------------------------------

# Start with an all-pass mask
high_pass_mask = np.ones((rows, cols, 2), np.float32)

# Create a circular region of zeros at the center
cv2.circle(
    high_pass_mask,
    (ccol, crow),
    LOW_PASS_RADIUS,
    (0, 0),
    -1
)

print("High-Pass Filter created.")

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 8. APPLY HIGH-PASS FILTER
# ------------------------------------------------------------

# Multiply DFT by High-Pass mask
high_pass_dft = dft_shift * high_pass_mask

# Shift frequency components back
high_pass_dft_shift = np.fft.ifftshift(high_pass_dft)

# Apply Inverse DFT
high_pass_result = cv2.idft(high_pass_dft_shift)

# Calculate magnitude
high_pass_result = cv2.magnitude(
    high_pass_result[:, :, 0],
    high_pass_result[:, :, 1]
)

# Normalize image
high_pass_result = cv2.normalize(
    high_pass_result,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

high_pass_result = np.uint8(high_pass_result)

print("High-Pass filtering completed.")


# ------------------------------------------------------------
# 9. SAVE OUTPUT IMAGES
# ------------------------------------------------------------

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "original.jpg"),
    image
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "magnitude_spectrum.jpg"),
    magnitude_spectrum
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "low_pass_filtered.jpg"),
    low_pass_result
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "high_pass_filtered.jpg"),
    high_pass_result
)

print("\nOutput images saved in:", OUTPUT_DIR)

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 10. DISPLAY RESULTS
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Magnitude Spectrum")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(low_pass_result, cmap="gray")
plt.title("Low-Pass Filtered Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(high_pass_result, cmap="gray")
plt.title("High-Pass Filtered Image")
plt.axis("off")

plt.tight_layout()

# Save combined result
plt.savefig(
    os.path.join(OUTPUT_DIR, "frequency_domain_results.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 11. DISPLAY FILTER MASKS
# ------------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(low_pass_mask[:, :, 0], cmap="gray")
plt.title("Low-Pass Filter Mask")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(high_pass_mask[:, :, 0], cmap="gray")
plt.title("High-Pass Filter Mask")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "filter_masks.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B

# ------------------------------------------------------------
# 12. FINAL MESSAGE
# ------------------------------------------------------------

print("\n==============================================")
print("EXPERIMENT 4 COMPLETED SUCCESSFULLY")
print("==============================================")
print("Generated files:")
print("1. original.jpg")
print("2. magnitude_spectrum.jpg")
print("3. low_pass_filtered.jpg")
print("4. high_pass_filtered.jpg")
print("5. frequency_domain_results.png")
print("6. filter_masks.png")
print("==============================================")

# Name      : PALLAV PANKAJ
# Roll No.  : 40
# Section   : B
 
# question and answer
# 
# # Experiment No. 4

## Frequency Domain Image Filtering using Fourier Transform

### 1. What is the Fourier Transform? Why is it important in digital image processing?

# The **Fourier Transform (FT)** is a mathematical technique that converts an image from the **spatial domain** into the **frequency domain**.

# In the spatial domain, an image is represented by pixel intensity values. In the frequency domain, the image is represented using different frequency components.

# Fourier Transform is important because it allows us to:

# * Analyze the frequency components of an image.
# * Separate low-frequency and high-frequency information.
# * Remove noise from images.
# * Enhance edges and fine details.
# * Perform image restoration and enhancement efficiently.

# ---

# ### 2. Differentiate between the spatial domain and the frequency domain.

# | Spatial Domain                                           | Frequency Domain                                      |
# | -------------------------------------------------------- | ----------------------------------------------------- |
# | Image is represented using pixel intensities.            | Image is represented using frequency components.      |
# | Operations are performed directly on pixels.             | Operations are performed on Fourier-transformed data. |
# | Suitable for simple local operations.                    | Suitable for frequency-based filtering.               |
# | Examples include smoothing and sharpening using kernels. | Examples include Low-Pass and High-Pass filtering.    |
# | Processing is based on spatial location.                 | Processing is based on frequency information.         |

# **In short:** Spatial-domain processing modifies pixels directly, while frequency-domain processing modifies the frequency components of an image.

# ---

# ### 3. What is the significance of the Discrete Fourier Transform (DFT) in image processing?

# The **Discrete Fourier Transform (DFT)** converts a digital image from the spatial domain into its frequency-domain representation.

# Its significance includes:

# * It identifies the low-frequency and high-frequency components of an image.
# * It helps analyze the frequency spectrum.
# * It enables frequency-domain filtering.
# * It can be used for noise removal and image enhancement.
# * It helps in edge and feature extraction.

# For digital images, the DFT is particularly useful because images contain discrete pixel values.

# ---

# ### 4. Explain the purpose of shifting the zero-frequency component to the center of the frequency spectrum.

# After applying the Fourier Transform, the **zero-frequency component** is normally located at the corners of the frequency spectrum.

# The `fftshift()` operation moves this component to the **center** of the spectrum.

# This makes the frequency spectrum easier to visualize and analyze:

# * **Center:** Low-frequency components.
# * **Away from center:** Higher-frequency components.
# * **Edges:** Highest-frequency components.

# Therefore, shifting the zero-frequency component to the center makes it easier to design and apply Low-Pass and High-Pass filters. The experiment specifically requires this shift before analyzing the magnitude spectrum.

# ---

# ### 5. Compare Low-Pass Frequency Filters and High-Pass Frequency Filters with suitable applications.

# | Low-Pass Filter (LPF)                       | High-Pass Filter (HPF)                                |
# | ------------------------------------------- | ----------------------------------------------------- |
# | Allows low frequencies to pass.             | Allows high frequencies to pass.                      |
# | Suppresses high-frequency components.       | Suppresses low-frequency components.                  |
# | Produces a smoother image.                  | Enhances edges and fine details.                      |
# | Helps reduce noise.                         | Helps highlight boundaries and fine structures.       |
# | Reduces sharp changes in intensity.         | Emphasizes sharp changes in intensity.                |
# | Application: noise reduction and smoothing. | Application: edge enhancement and feature extraction. |

# According to the experiment, the Low-Pass Filter is used to reduce image noise, while the High-Pass Filter is used to enhance edges and fine details.

# ---

# ### 6. What is the role of the Inverse Fourier Transform (IDFT) in image reconstruction?

# The **Inverse Discrete Fourier Transform (IDFT)** converts a filtered image from the frequency domain back into the spatial domain.

# The process is:

# **Original Image → DFT → Frequency Filtering → IDFT → Reconstructed Image**

# After applying a frequency filter, the resulting frequency-domain data cannot be directly viewed as a normal image. IDFT reconstructs the filtered image so that it can be displayed and analyzed.

# The experiment uses IDFT to reconstruct both Low-Pass and High-Pass filtered images.

# ---

# ### 7. Why is frequency domain filtering preferred for certain image enhancement tasks?

# Frequency-domain filtering is preferred for certain tasks because it provides direct control over different frequency components of an image.

# Advantages include:

# 1. **Selective filtering** – Specific frequency ranges can be removed or preserved.
# 2. **Noise reduction** – High-frequency noise can be suppressed using Low-Pass Filters.
# 3. **Edge enhancement** – High-frequency components can be emphasized using High-Pass Filters.
# 4. **Efficient processing** – Fourier-based techniques can be efficient for large images and complex filtering operations.
# 5. **Better frequency analysis** – The frequency spectrum provides useful information about image characteristics.

# It is particularly useful for image enhancement, restoration, and feature extraction.

# ---

# ### 8. Mention four real-world applications where Fourier Transform is used in computer vision and image analysis.

# Four real-world applications are:

# 1. **Medical Image Enhancement** – Used to improve and analyze medical images.
# 2. **Satellite Image Analysis** – Used for analyzing frequency characteristics of satellite imagery.
# 3. **Image Restoration** – Used to remove noise and recover image information.
# 4. **Biometric Systems** – Used in processing and analyzing biometric images such as fingerprints and facial images.

# These applications are specifically mentioned in the experiment description.

# ---

# ### 9. Compare frequency domain filtering with spatial domain filtering based on computational efficiency and practical applications.

# | Frequency Domain Filtering                                    | Spatial Domain Filtering                                   |
# | ------------------------------------------------------------- | ---------------------------------------------------------- |
# | Converts the image to the frequency domain before filtering.  | Works directly on image pixels.                            |
# | Uses Fourier Transform and Inverse Fourier Transform.         | Usually uses masks/kernels directly on pixels.             |
# | Can be efficient for large and complex filtering operations.  | Simple and effective for local filtering operations.       |
# | Provides direct control over frequency components.            | Provides direct control over neighboring pixels.           |
# | Useful for image restoration and frequency-based enhancement. | Useful for smoothing, sharpening, and basic noise removal. |
# | More complex to implement.                                    | Relatively simple to implement.                            |

# Therefore, spatial filtering is often convenient for simple local operations, whereas frequency-domain filtering is useful when specific frequency components need to be manipulated.

# ---

# ### 10. How does frequency domain filtering improve the performance of image restoration and feature extraction techniques?

# Frequency-domain filtering improves image processing by selectively manipulating the frequency components of an image.

# For **image restoration**:

# * Low-frequency or high-frequency components can be selectively preserved or suppressed.
# * High-frequency noise can be reduced using Low-Pass Filters.
# * Image quality can be improved by removing unwanted frequency components.

# For **feature extraction**:

# * High-Pass Filters emphasize edges and fine details.
# * Important boundaries and structures become more prominent.
# * Enhanced edges can make subsequent feature extraction easier.

# Thus, frequency-domain filtering can improve the quality of restored images and make important features more distinguishable. The experiment specifically evaluates its effects on smoothness, edge enhancement, noise removal, and feature preservation.

# # Conclusion

# Frequency-domain image filtering uses the **Fourier Transform** to represent an image in terms of its frequency components. **Low-Pass Filters** are useful for smoothing and noise reduction, while **High-Pass Filters** enhance edges and fine details. The **Inverse Fourier Transform** converts the filtered frequency representation back into a spatial-domain image. These techniques are useful in image enhancement, restoration, medical imaging, satellite image analysis, and biometric systems.
 