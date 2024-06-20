# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import os
import cv2
import numpy as np

# termination criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

checker = (5, 6)  # (rows, columns)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1, checker[0] * checker[1], 3), np.float32)  # 3D points
objp[0, :, :2] = np.mgrid[0 : checker[0], 0 : checker[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# images = glob.glob('./Images/*.jpg')
images = os.listdir("./Images")

for i in range(len(images)):
    img = cv2.imread("./Images/" + images[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, checker, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, checker, corners2, ret)
        # cv2.namedWindow("image" + str(i), cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("image" + str(i), 800, 600)
        # cv2.imshow("image" + str(i), img)

## Calibration
# cv2.calibrateCamera() which returns the camera matrix, distortion coefficients, rotation and translation vectors etc.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

## Undistortion
# first, we can refine the camera matrix based on a free scaling parameter using cv.getOptimalNewCameraMatrix()
# If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels. So it may even remove some pixels at image corners. If alpha=1, all pixels are retained with some extra black images. This function also returns an image ROI which can be used to crop the result.
img = cv2.imread("./Images/" + images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# 1. Using cv2.undistort()
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]
cv2.namedWindow("Undistorted Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Undistorted Image", 800, 600)
cv2.imshow("Undistorted Image", dst)

img = cv2.imread("./Images/" + images[0])
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", 800, 600)
cv2.imshow("Original Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Using remapping
# This is the most accurate method. We can find the map for remapping using cv2.initUndistortRectifyMap() and cv2.remap() functions.
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]
cv2.namedWindow("Undistorted Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Undistorted Image", 800, 600)
cv2.imshow("Undistorted Image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
