import cv2
import numpy as np

# 체커보드 패턴 크기 (가로 칸 수, 세로 칸 수)
CHECKERBOARD = (6, 9)

# 이미지 로드
image = cv2.imread("data/calibration_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 체커보드 감지
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if ret:
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (int(x), int(y)), 5, (0,0,255), -1)

    cv2.imwrite("data/chessboard_detected.jpg", image)

    # 저장 (u, v)
    np.savetxt("data/image_points.txt", corners.reshape(-1, 2), fmt="%.2f")
