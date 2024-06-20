import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread("./hobanwu2.jpg", cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread("./image/img (14).jpg", cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(
    img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.imshow(img3), plt.show()


# Assuming kp2 is a tuple containing cv2.KeyPoint objects
decimal_points = []

for kp in kp2:
    x = kp.pt[0]  # Extract x-coordinate
    y = kp.pt[1]  # Extract y-coordinate
    decimal_points.append((x, y))  # Append (x, y) tuple to the list

# Print the list of decimal points
decimal_points = np.array(decimal_points)
for i in range(len(decimal_points)):
    print(decimal_points[i])
