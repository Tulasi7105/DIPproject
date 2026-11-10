

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load road image
image = cv2.imread(r"C:\Users\tulasi\Downloads\shawn-mcnulty-berMrRj8wPI-unsplash.jpg")

if image is None:
    print("Error: Image not found. Check the image path.")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Remove noise using Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Improve image contrast
equalized = cv2.equalizeHist(blur)

# Step 5: Detect edges using Canny
edges = cv2.Canny(equalized, 50, 150)

# Step 6: Apply thresholding
_, threshold = cv2.threshold(
    equalized, 100, 255, cv2.THRESH_BINARY_INV
)

# Step 7: Morphological operation
kernel = np.ones((5, 5), np.uint8)

morphology = cv2.morphologyEx(
    threshold, cv2.MORPH_CLOSE, kernel
)

# Step 8: Find contours
contours, _ = cv2.findContours(
    morphology,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Step 9: Detect potential damaged regions
result = image_rgb.copy()

damage_count = 0

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 100:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        damage_count += 1

# Step 10: Display results
images = [
    image_rgb,
    gray,
    blur,
    equalized,
    edges,
    morphology,
    result
]

titles = [
    "Original Image",
    "Grayscale",
    "Noise Removal",
    "Contrast Enhancement",
    "Edge Detection",
    "Morphological Processing",
    "Potential Damage Detection"
]

plt.figure(figsize=(15, 10))

for i in range(len(images)):
    plt.subplot(3, 3, i + 1)

    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap="gray")
    else:
        plt.imshow(images[i])

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# Step 11: Display detection count
print("Road Damage Detection Completed")
print("Potential damaged regions:", damage_count)

# Step 12: Save output
cv2.imwrite(
    "dataset/output/detected_damage.jpg",
    cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
)

print("Output image saved successfully!")