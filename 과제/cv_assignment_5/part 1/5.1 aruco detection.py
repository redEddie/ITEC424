import cv2
from matplotlib import pyplot as plt


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


def main():
    # read world coordinates of markers
    file_path = "./asset/markermapID.txt"
    data = read_marker_data(file_path)

    # path to the image files
    image_path = "./images/KakaoTalk_20240530_125615705_17.jpg"

    # detect markers in the images
    image, corners, ids = detect_markers(image_path)

    # draw markers in the images
    draw_markers(image, corners, ids, data)


if __name__ == "__main__":
    main()
