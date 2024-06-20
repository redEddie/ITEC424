def read_text_file(file_path):
    data = {}
    with open(file_path, "r") as file:
        # 파일의 각 줄을 읽음
        lines = file.readlines()
        current_id = None
        for line in lines:
            line = line.strip()
            if line:
                # ArUco 마커의 ID를 읽고,
                if line.isdigit():
                    current_id = line.strip()
                    data[current_id] = []
                # 해당 ID의 data에 points(x,y,z)를 입력함.
                else:
                    points = [float(val) for val in line.split()[:3]]
                    data[current_id].append(points)
    # 결과값 data는 각 ArUco ID와 그에 해당되는 4개의 코너점들을 가지고 있음.
    return data


text_data = read_text_file("./asset/markermapID.txt")

# %%
import cv2  # AttributeError: module 'cv2.aruco' has no attribute 'Dictionary_get' >> pip install opencv-contrib-python==4.6.0.66


def detect_aruco(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #'DICT_6X6_250' : ArUco 생성 시 설정하는 파라미터, 사용자의 환경에 맞게 수정가능함.
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    # 흑백영상으로부터 ArUco 마커 검출, 결과값은 ID와 각 corner 4개 점들
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return ids, corners


detected_ids, detected_corners = detect_aruco(
    "./images/KakaoTalk_20240530_125615705_01.jpg"
)

# %%


def compare_ids(text_data, detected_ids):
    matched_ids = []
    matched_points = []
    detected_ids_list = detected_ids.flatten().tolist()
    for id, points in text_data.items():
        if str(id) in map(str, detected_ids_list):
            matched_ids.append(id)
            matched_points.append(points)
    return matched_ids, matched_points


matched_ids, matched_points = compare_ids(text_data, detected_ids)


# %%

import numpy as np

# PnP알고리즘을 돌리기 위해, 데이터 사전정렬
target_updated_corners = []
updated_3dpoint = []
for id_str, corner_points in zip(matched_ids, matched_points):
    # Find the index of the ID in detected_ids
    id_index = np.where(detected_ids == int(id_str))[1][0]
    target_updated_corners.append(detected_corners[id_index][0])
    updated_3dpoint.append(corner_points)

selected_target_updated_corners = [
    point for sublist in target_updated_corners for point in sublist
]
selected_updated_3dpoint = [point for sublist in updated_3dpoint for point in sublist]
# %%
selected_target_updated_corners = np.array(selected_target_updated_corners)
selected_updated_3dpoint = np.array(selected_updated_3dpoint)

# %%

import cv2
import numpy as np

# 카메라 행렬 (과제 2에서 구한 값으로 대치)
cameraMatrix = np.array(
    [[3.28261164e3, 0, 2.02403056e3], [0, 3.28885353e3, 1.45308413e3], [0, 0, 1]]
)
distCoeffs = np.array(
    [2.287579407e-1, -1.52709073e0, -6.24107384e-3, 2.07482576e-3, 2.337278047e0]
)

# Solve PnP
# success, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
success, rvec, tvec = cv2.solvePnP(
    selected_updated_3dpoint, selected_target_updated_corners, cameraMatrix, distCoeffs
)
if success:
    print("Rotation Vector:\n", rvec)
    print("Translation Vector:\n", tvec)
else:
    print("solvePnP failed")

# %%
R, _ = cv2.Rodrigues(rvec)
# 최종 결과(Transformation Matrix Visualization)
T = tvec
print(np.hstack((R, T)))

# 파일로 저장
np.savetxt("pnpresult.txt", np.hstack((R, T)), fmt="%.8f")
