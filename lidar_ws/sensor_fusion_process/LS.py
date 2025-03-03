import numpy as np
import cv2

# 📌 데이터 불러오기
data = np.loadtxt("data/data.txt")  # data.txt 파일 경로

# LiDAR 좌표 (x, y)
lidar_points = data[:, :2]

# 카메라 좌표 (u, v)
image_points = data[:, 2:]

# 📌 Homography 행렬 계산 (2D 변환)
H, _ = cv2.findHomography(lidar_points, image_points, cv2.RANSAC)

# 📌 변환 행렬 출력
print("📌 Homography Matrix (LiDAR → Camera 변환)")
print(H)

# 📌 변환 행렬을 파일로 저장
np.savetxt("data/H_matrix.txt", H, fmt="%.6f")
print("변환 행렬이 'H_matrix.txt' 파일에 저장되었습니다!")
