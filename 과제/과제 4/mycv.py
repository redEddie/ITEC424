import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def show_img(jpg):
    gray_image = cv.cvtColor(jpg, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(gray_image, 127 + 10, 255, cv.THRESH_BINARY)

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(jpg, cv.COLOR_BGR2RGB))
    plt.title("Original Image")

    # Grayscale image
    plt.subplot(1, 3, 2)
    plt.imshow(gray_image, cmap="gray")
    plt.title("Grayscale Image")

    # Binary image
    plt.subplot(1, 3, 3)
    plt.imshow(binary_image, cmap="binary")
    plt.title("Binary Image")

    plt.tight_layout()
    plt.show()

    return gray_image, binary_image


jpg1 = cv.imread("./KakaoTalk_20240516_134041401.jpg")
jpg2 = cv.imread("./KakaoTalk_20240516_134148374.jpg")

gray1, bi1 = show_img(jpg1)
gray2, bi2 = show_img(jpg2)


def new_func(image, binary):
    A4_SIZE_X = 297 * 3
    A4_SIZE_Y = 210 * 3
    dst = np.array(
        [
            [0, A4_SIZE_Y - 1],
            [0, 0],
            [A4_SIZE_X - 1, 0],
            [A4_SIZE_X - 1, A4_SIZE_Y - 1],
        ],
        dtype=np.float32,
    )

    # Step 1: 윤곽선을 찾습니다.
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)

    # Step 2: 직사각형을 찾습니다.
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = box.astype(int)

    # Step 3: 사전에 주어진 frame 크기에 맞추기 위해 변환 행렬을 계산합니다.
    perspective_transform = cv.getPerspectiveTransform(box.astype(np.float32), dst)

    # Step 4: 원근 변환을 수행합니다.
    warped_image = cv.warpPerspective(image, perspective_transform, (297 * 3, 210 * 3))

    # 결과를 표시합니다.
    cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
    cv.resizeWindow("Original Image", 800, 600)
    cv.imshow("Original Image", image)
    cv.namedWindow("Document Warp", cv.WINDOW_NORMAL)
    cv.resizeWindow("Document Warp", 297 * 3, 210 * 3)
    cv.imshow("Document Warp", warped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


new_func(jpg1, bi1)
new_func(jpg2, bi2)
