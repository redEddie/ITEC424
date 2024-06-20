import os
import cv2

image_dir = "./새로운Images"
images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]


for i in range(len(images)):
    image_path = os.path.join(image_dir, images[i])

    # src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.namedWindow(f"image {i}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"image {i}", 400, 300)
    # cv2.imshow(f"image {i}", images[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
