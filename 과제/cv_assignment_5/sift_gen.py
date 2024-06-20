import numpy as np
from matplotlib import pyplot as plt
import os
import cv2


def detect_markers(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return image, corners, ids


def find_match_points(corners_1, ids_1, corners_2, ids_2):
    matching_ids = []
    matching_idx_1 = []
    matching_idx_2 = []
    matching_corners_1 = []
    matching_corners_2 = []

    for i in range(len(ids_1)):
        for j in range(len(ids_2)):
            if ids_1[i] == ids_2[j]:
                matching_ids.append(ids_1[i])

                matching_corners_1.append(corners_1[i].reshape(-1, 2))
                matching_corners_2.append(corners_2[j].reshape(-1, 2))

                matching_idx_1.append((4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
                matching_idx_2.append((4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3))

    # Now we have the matching IDs and corresponding corner points
    # You can use these lists for further processing with the 5-point algorithm
    matching_corners_1 = np.array(matching_corners_1).reshape(-1, 2)
    matching_corners_2 = np.array(matching_corners_2).reshape(-1, 2)

    return (
        matching_ids,
        matching_idx_1,
        matching_idx_2,
        matching_corners_1,
        matching_corners_2,
    )


def generate_sift_files(image, image_dir):
    image_path = os.path.join(image_dir, image)
    image, corners, ids = detect_markers(image_path)

    _image = image_path.split(".")[1]

    with open(f".{_image}.sift", "w") as output:
        output.write(f"{len(corners)*4} 128\n")

        image, corners, ids = detect_markers(image_path)
        for i, corner in enumerate(corners):
            for j, c in enumerate(corner[0]):
                output.write(f"{c[0]} {c[1]} 0 0\n")
                output.write(
                    "0 " * 128 + "\n"
                )  # Write the 128 descriptor values (all zeros in this case)

                # Check if this is not the last element
                # if not (i == len(corners) - 1 and j == len(corner[0]) - 1):
                # output.write("\n")
        output.close()


def write_match_file(target_1, target_2, idx1, idx2, image_dir):
    image_path = os.path.join(image_dir, target_1)
    image_1, corners_1, ids_1 = detect_markers(image_path)
    image_path = os.path.join(image_dir, target_2)
    image_2, corners_2, ids_2 = detect_markers(image_path)
    (
        _matching_ids,
        matching_idx_1,
        matching_idx_2,
        matching_corners_1,
        matching_corners_2,
    ) = find_match_points(corners_1, ids_1, corners_2, ids_2)

    if _matching_ids == []:
        return

    current_dir = os.getcwd()
    image_1_dir = os.path.join(current_dir, "images", target_1)
    image_2_dir = os.path.join(current_dir, "images", target_2)

    # Ensure the drive letter is capitalized
    image_1_dir = image_1_dir[0].upper() + image_1_dir[1:]
    image_2_dir = image_2_dir[0].upper() + image_2_dir[1:]

    with open(f"./matches/match_{idx1}_{idx2}.txt", "w") as output:
        output.write(f"{image_1_dir}\n")
        output.write(f"{image_2_dir}\n")
        output.write(f"{len(matching_corners_1)}\n")
        output.write(
            " ".join(
                map(
                    str,
                    np.array(matching_idx_1).reshape(
                        -1,
                    ),
                )
            )
            + "\n"
        )
        output.write(
            " ".join(
                map(
                    str,
                    np.array(matching_idx_2).reshape(
                        -1,
                    ),
                )
            )
            + "\n"
        )

        output.close()


# 시작
image_dir = "./images"
images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

## sift
for image in images:
    generate_sift_files(
        image=image,
        image_dir=image_dir,
    )

## matches
# Create 'matches' folder if it doesn't exist
matches_dir = os.path.join(image_dir, "matches")
if not os.path.exists(matches_dir):
    os.makedirs(matches_dir)

for index1, image1 in enumerate(images):
    for index2, image2 in enumerate(images):
        if index1 == index2:
            continue
        write_match_file(image1, image2, index1, index2, image_dir)
