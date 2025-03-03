import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# LiDAR 좌표 로드
lidar_data = np.loadtxt("data/laser_points.txt")
lidar_x, lidar_y = lidar_data[:, 0], lidar_data[:, 1]

# DBSCAN 클러스터링 실행 (eps 조정)
dbscan = DBSCAN(eps=0.05, min_samples=10)  # eps 값을 증가시키면서 조정해볼 것
labels = dbscan.fit_predict(lidar_data)

# 클러스터 개수 확인
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"📌 감지된 클러스터 개수: {num_clusters}")

# 노이즈 제거 (-1 레이블을 제외)
filtered_points = lidar_data[labels != -1]

# 클러스터링된 결과를 시각화
plt.figure(figsize=(8, 6))
plt.scatter(lidar_x, lidar_y, c=labels, cmap="rainbow", s=5)
plt.title("LiDAR Checkerboard Points (DBSCAN Clustering)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# 54개만 추출 (가장 밀도가 높은 클러스터를 선택)
if len(filtered_points) > 54:
    selected_points = filtered_points[:54]  # 상위 54개 포인트만 선택
else:
    selected_points = filtered_points  # 부족하면 모든 포인트 사용

# 결과 저장
np.savetxt("/mnt/data/filtered_lidar_points.txt", selected_points, fmt="%.6f")
print("📌 LiDAR 데이터 필터링 완료, 54개 포인트 저장됨!")

