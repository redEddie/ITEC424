import cv2
import numpy as np


def read_text_file(file_path):
    data = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        current_id = None
        for line in lines:
            line = line.strip()
            if line.isdigit():
                current_id = line.strip()
                data[current_id] = []
            else:
                points = [float(val) for val in line.split()[:3]]
                data[current_id].append(points)
    return data


def detect_aruco(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return ids, corners


def compare_ids(text_data, detected_ids):
    matched_ids = []
    matched_points = []
    detected_ids_list = detected_ids.flatten().tolist()
    for id, points in text_data.items():
        if str(id) in map(str, detected_ids_list):
            matched_ids.append(id)
            matched_points.append(points)
    return matched_ids, matched_points


def solve_pnp(selected_updated_3dpoint, selected_target_updated_corners):
    cameraMatrix = np.array(
        [[3.28261164e3, 0, 2.02403056e3], [0, 3.28885353e3, 1.45308413e3], [0, 0, 1]]
    )
    distCoeffs = np.array(
        [2.287579407e-1, -1.52709073e0, -6.24107384e-3, 2.07482576e-3, 2.337278047e0]
    )
    success, rvec, tvec = cv2.solvePnP(
        selected_updated_3dpoint,
        selected_target_updated_corners,
        cameraMatrix,
        distCoeffs,
    )
    if success:
        print("Rotation Vector:\n", rvec)
        print("Translation Vector:\n", tvec)
    else:
        print("solvePnP failed")
    return rvec, tvec


def save_result(R, T):
    np.savetxt("pnpresult.txt", np.hstack((R, T)), fmt="%.8f")


def main():
    text_data = read_text_file("./asset/markermapID.txt")
    detected_ids, detected_corners = detect_aruco(
        "./images/KakaoTalk_20240530_125615705_01.jpg"
    )
    matched_ids, matched_points = compare_ids(text_data, detected_ids)
    target_updated_corners = []
    updated_3dpoint = []
    for id_str, corner_points in zip(matched_ids, matched_points):
        id_index = np.where(detected_ids == int(id_str))[1][0]
        target_updated_corners.append(detected_corners[id_index][0])
        updated_3dpoint.append(corner_points)
    selected_target_updated_corners = [
        point for sublist in target_updated_corners for point in sublist
    ]
    selected_updated_3dpoint = [
        point for sublist in updated_3dpoint for point in sublist
    ]
    selected_target_updated_corners = np.array(selected_target_updated_corners)
    selected_updated_3dpoint = np.array(selected_updated_3dpoint)
    R, T = solve_pnp(selected_updated_3dpoint, selected_target_updated_corners)
    R, _ = cv2.Rodrigues(R)
    save_result(R, T)


if __name__ == "__main__":
    main()
