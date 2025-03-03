import numpy as np

# LiDAR 포인트 로드
lidar_points = np.loadtxt("data/laser_points.txt")
# 카메라 포인트 로드
image_points = np.loadtxt("data/image_points.txt")

# 두 포인트 개수 확인
assert lidar_points.shape[0] == image_points.shape[0], "데이터 개수가 일치하지 않습니다!"

# 데이터 결합
data = np.hstack((lidar_points, image_points))  # x, y, u, v 순서로 결합

# 저장
np.savetxt("data/data.txt", data, fmt="%.6f")

print("Data saved successfully!")

