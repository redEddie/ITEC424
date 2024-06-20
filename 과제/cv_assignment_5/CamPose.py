import cv2 as cv
import numpy as np
import glob

path_output = 'C:/pythoncode/ANEG/'
path_input =  'C:/pythoncode/ANEG/'
path_extrinsic  = glob.glob(f'{path_input}/*.txt')
size_camera = 0.15
num_camera = len(path_extrinsic)

points = []
for file_name in path_extrinsic :
    file = open(file_name)
    data = file.read().splitlines()
    file.close()

    RT = np.array([np.array(data[0].split(" "),dtype=np.float32),
                   np.array(data[1].split(" "),dtype=np.float32),
                   np.array(data[2].split(" "),dtype=np.float32),
                   [0,0,0,1]])
    points.append(np.matmul(np.linalg.inv(RT), np.array([0, 0, 0, 1])))
    points.append(np.matmul(np.linalg.inv(RT), np.array([-size_camera,-size_camera, size_camera, 1])))
    points.append(np.matmul(np.linalg.inv(RT), np.array([-size_camera, size_camera, size_camera, 1])))
    points.append(np.matmul(np.linalg.inv(RT), np.array([size_camera, -size_camera, size_camera, 1])))
    points.append(np.matmul(np.linalg.inv(RT), np.array([size_camera,  size_camera, size_camera, 1])))
points = np.array(points)

f = open(f'{path_output}/camera.obj','w')  
f.write(f'####\n# Object camera.obj\n#\n# Vertices: {num_camera*5}\n# Faces: 0\n#\n####\n')
for i in range(num_camera*5):
    f.write(f'vn 0.0000 0.0000 0.0000\n')
    f.write(f'v {points[i,0]} {points[i,1]} {points[i,2]}\n')
f.write(f'# {num_camera*5} vertices, 0 vertices normals\n\n')
for i in range(num_camera):
    f.write(f'l {5*i+1} {5*i+2}\n')
    f.write(f'l {5*i+1} {5*i+3}\n')
    f.write(f'l {5*i+1} {5*i+4}\n')
    f.write(f'l {5*i+1} {5*i+5}\n')
    f.write(f'l {5*i+2} {5*i+3}\n')
    f.write(f'l {5*i+3} {5*i+5}\n')
    f.write(f'l {5*i+5} {5*i+4}\n')
    f.write(f'l {5*i+4} {5*i+2}\n')
f.write(f'# 0 faces, 0 coords texture\n\n# End of File')
f.close()
