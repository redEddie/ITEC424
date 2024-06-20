import cv2
from matplotlib import pyplot as plt
import numpy as np


def read_marker_data(file_path):
    data = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        current_id = None
        for line in lines:
            line = line.strip()
            if line:
                if line.isdigit():
                    current_id = line.strip()
                    data[current_id] = []
                else:
                    points = [float(val) for val in line.split()[:3]]
                    data[current_id].append(points)
    return data


def detect_markers(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return image, corners, ids


def draw_markers(image, corners, ids, data):
    plt.figure(figsize=(12, 8))
    if len(corners) > 0:
        ids = ids.flatten()

        for marker_corner, marker_id in zip(corners, ids):
            corner = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corner

            top_left = (int(top_left[0]), int(top_left[1]))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

            x_values = [
                top_left[0],
                top_right[0],
                bottom_right[0],
                bottom_left[0],
                top_left[0],
            ]
            y_values = [
                top_left[1],
                top_right[1],
                bottom_right[1],
                bottom_left[1],
                top_left[1],
            ]
            plt.plot(x_values, y_values, color="red", linewidth=0.5)

            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)
            plt.plot(cX, cY, "ro", markersize=1)

            if str(marker_id) in data:
                (x, y, z) = data[str(marker_id)][0]
                plt.text(
                    top_left[0],
                    top_left[1] - 25,
                    f"ID: {marker_id}",
                    fontsize=8,
                    color="green",
                )
                plt.text(
                    top_left[0],
                    top_left[1] - 100,
                    f"X: {x:.2f}",
                    fontsize=8,
                    color="green",
                )
                plt.text(
                    top_left[0],
                    top_left[1] - 175,
                    f"Y: {y:.2f}",
                    fontsize=8,
                    color="green",
                )
                plt.text(
                    top_left[0],
                    top_left[1] - 250,
                    f"Z: {z:.2f}",
                    fontsize=8,
                    color="green",
                )

                plt.plot(top_left[0], top_left[1], "go", markersize=2)

        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()


def estimate_pose_5point(corners_1, ids_1, corners_2, ids_2, K):
    matching_ids = []
    matching_corners_1 = []
    matching_corners_2 = []

    for i in range(len(ids_1)):
        for j in range(len(ids_2)):
            if ids_1[i] == ids_2[j]:
                matching_ids.append(ids_1[i])
                matching_corners_1.append(corners_1[i].reshape(-1, 2))
                matching_corners_2.append(corners_2[j].reshape(-1, 2))

    # Now we have the matching IDs and corresponding corner points
    # You can use these lists for further processing with the 5-point algorithm
    matching_corners_1 = np.array(matching_corners_1).reshape(-1, 2)
    matching_corners_2 = np.array(matching_corners_2).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        matching_corners_1,
        matching_corners_2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    _, R, t, mask = cv2.recoverPose(
        E, matching_corners_1, matching_corners_2, cameraMatrix=K
    )

    return E, R, t


def compute_distance(K, R, t):
    # 카메라 1과 카메라 2의 내부 파라미터와 회전 행렬, 변환 벡터를 받아옵니다.
    # K는 내부 파라미터 행렬, R은 회전 행렬, t는 변환 벡터입니다.

    # 카메라 1의 광학 중심을 원점으로 설정합니다.
    c1 = np.array([0, 0, 0])

    # 카메라 2의 광학 중심을 t로 설정합니다.
    c2 = -np.dot(np.linalg.inv(R), t)

    # 두 광학 중심 사이의 거리를 계산합니다.
    distance = np.linalg.norm(c2 - c1)

    return distance


def main():
    # read world coordinates of markers
    file_path = "./asset/markermapID.txt"
    data = read_marker_data(file_path)

    # path to the image files
    image_file_1 = "./images/KakaoTalk_20240530_125615705_01.jpg"
    image_file_2 = "./images/KakaoTalk_20240530_125615705_03.jpg"

    # detect markers in the images
    image_1, corners_1, ids_1 = detect_markers(image_file_1)
    image_2, corners_2, ids_2 = detect_markers(image_file_2)

    # camera matrix
    K = np.array(
        [[3.28261164e3, 0, 2.02403056e3], [0, 3.28885353e3, 1.45308413e3], [0, 0, 1]]
    )

    E, R, t = estimate_pose_5point(corners_1, ids_1, corners_2, ids_2, K)

    # 두 카메라 사이의 거리를 계산합니다.
    distance = compute_distance(K, R, t)

    print("Estimated Essential Matrix:\n", E)
    print("Recovered Rotation Matrix:\n", R)
    print("Recovered Translation Vector:\n", t)
    print("두 카메라 사이의 거리:", distance)


if __name__ == "__main__":
    main()
