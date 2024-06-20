import os
import cv2

images = os.listdir("./Images")

for i in range(len(images)):
    src = cv2.imread("./Images/" + images[i], cv2.IMREAD_COLOR)
    cv2.namedWindow("image" + str(i), cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image" + str(i), 400, 300)
    cv2.imshow("image" + str(i), src)

cv2.waitKey(0)
cv2.destroyAllWindows()

# src0 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src1 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src2 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src3 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src4 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src5 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src6 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src7 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src8 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# src9 = cv2.imread("./Images/IMG_9726.jpg", cv2.IMREAD_COLOR)
# cv2.namedWindow("image0", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("image0", 400, 300)
# cv2.imshow("image0", src0)
# cv2.namedWindow("image1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("image1", 400, 300)
# cv2.imshow("image1", src1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
