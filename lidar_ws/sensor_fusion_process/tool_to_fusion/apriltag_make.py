import cv2
import numpy as np
import os

# ========== 설정값 ==========
TAG_SIZE_MM = 55  # 개별 AprilTag 크기 (mm)
TAG_SPACE_MM = 25  # 태그 간격 (mm)
TAGS_PER_ROW = 5  # 한 행에 들어가는 태그 수
TAGS_PER_COL = 5  # 한 열에 들어가는 태그 수
DPI = 300  # 해상도 (dots per inch)
TAG_IMAGE_FOLDER = "tag36h11"  # AprilTag 이미지가 저장된 폴더
OUTPUT_IMAGE_NAME = "apriltag_5x5_grid.png"

# ========== 픽셀 변환 ==========
MM_TO_INCH = 1 / 25.4  # mm를 inch로 변환
PIXELS_PER_MM = DPI * MM_TO_INCH  # 1mm당 픽셀 수

TAG_SIZE_PX = int(TAG_SIZE_MM * PIXELS_PER_MM)  # 태그 크기 (픽셀)
TAG_SPACE_PX = int(TAG_SPACE_MM * PIXELS_PER_MM)  # 태그 간격 (픽셀)

GRID_WIDTH = (TAG_SIZE_PX * TAGS_PER_ROW) + (TAG_SPACE_PX * (TAGS_PER_ROW - 1))
GRID_HEIGHT = (TAG_SIZE_PX * TAGS_PER_COL) + (TAG_SPACE_PX * (TAGS_PER_COL - 1))

# 빈 캔버스 생성 (흰 배경)
grid_image = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8) * 255

# ========== 5x5 태그 배치 ==========
for row in range(TAGS_PER_COL):
    for col in range(TAGS_PER_ROW):
        tag_id = row * TAGS_PER_ROW + col  # 0~24까지 ID 사용
        tag_filename = f"apriltag-imgs/tag36h11/tag36_11_{tag_id:05d}.png"
        
        # 태그 이미지 로드
        if not os.path.exists(tag_filename):
            print(f"파일 없음: {tag_filename}")
            continue
        
        tag_image = cv2.imread(tag_filename, cv2.IMREAD_GRAYSCALE)
        if tag_image is None:
            print(f"이미지를 불러올 수 없음: {tag_filename}")
            continue
        
        # 태그 크기 조정
        tag_image = cv2.resize(tag_image, (TAG_SIZE_PX, TAG_SIZE_PX), interpolation=cv2.INTER_NEAREST)
        
        # 태그 위치 계산
        x_offset = col * (TAG_SIZE_PX + TAG_SPACE_PX)
        y_offset = row * (TAG_SIZE_PX + TAG_SPACE_PX)

        # 태그 배치
        grid_image[y_offset:y_offset + TAG_SIZE_PX, x_offset:x_offset + TAG_SIZE_PX] = tag_image

# ========== 결과 저장 ==========
cv2.imwrite(OUTPUT_IMAGE_NAME, grid_image)
print(f"5x5 AprilTag Grid 이미지 저장 완료: {OUTPUT_IMAGE_NAME}")

# 이미지 확인
cv2.imshow("AprilTag Grid", grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

