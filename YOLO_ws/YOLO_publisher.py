#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
import torch
from ultralytics import YOLO
from sensor_msgs.msg import LaserScan, CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from cv_bridge import CvBridge
import time
from scipy.spatial import KDTree
from math import cos, sin

# 학습데이터 정보
con = 0  # 라바콘
drum = 1  # PE드럼

# con 거슬려서 cone으로 바꿈
CLASS_MAPPING = {
    con: "cone",
    drum: "drum"
}

# 하이퍼파라미터: 바운딩 박스에서 사용할 높이 비율 (0.0 = 아래, 1.0 = 꼭대기)
HEIGHT_RATIO = 0.3  # 0.0 ~ 1.0

class ConeDrumDetection:
    def __init__(self):
        rospy.init_node('cone_drum_detector', anonymous=True)

        # ROS 파라미터에서도 높이 비율 설정 가능하게 해둠
        self.height_ratio = rospy.get_param("~height_ratio", HEIGHT_RATIO)

        # 퍼블리셔
        self.result_publish = rospy.Publisher("/detection_result", String, queue_size=10)
        self.visualization_publish = rospy.Publisher("/yolo_viz", MarkerArray, queue_size=10)

        # 섭스크라이버
        ################ 이미지 토픽 뭔지 확인하기
        # self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback, queue_size=10)
        self.lidar_sub = rospy.Subscriber("/point_cloud", LaserScan, self.lidar_callback)

        # YOLO 모델 로드
        self.model = YOLO('/home/user/YOLO_ws/weights/YOLO_0216.pt') ################ 경로 수정하기

        # 이미지 및 LiDAR 데이터 저장 변수
        self.bridge = CvBridge()
        self.img_bgr = None
        self.lidar_points = None
        self.filtered_points = None

        # YOLO 실행 간격 (프레임 간격 조정)
        self.frame_counter = 0
        self.frame_interval = 2  # 2 프레임마다 YOLO 실행 (연산량 조절)

    def image_callback(self, msg):
        """ 카메라 이미지 수신 후 YOLO 감지 수행 """
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def lidar_callback(self, msg):
        """ 2D LiDAR 데이터를 (x, y) 좌표로 변환 후 ROI 필터 적용 """
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # 극좌표 → 직교좌표 변환
        self.lidar_points = np.array([ranges * np.cos(angles), ranges * np.sin(angles)]).T

        # ROI 필터 적용 (전방 10m 이내, 좌우 ±3m 범위)
        roi_mask = (self.lidar_points[:, 0] > 0.5) & (self.lidar_points[:, 0] < 10.0) & \
                   (self.lidar_points[:, 1] > -3.0) & (self.lidar_points[:, 1] < 3.0)
        self.filtered_points = self.lidar_points[roi_mask]

    def process_detections(self):
        """ YOLO 탐지 수행 후 2D LiDAR와 매칭 """
        if self.img_bgr is not None:
            self.frame_counter += 1
            if self.frame_counter % self.frame_interval == 0:
                res = self.model.predict(self.img_bgr)
                plots = res[0].plot()

                markers = MarkerArray()
                detected_classes = set()

                if len(res[0].boxes) > 0:
                    for i, box in enumerate(res[0].boxes):
                        cls = int(box.cls.item())  # YOLO가 예측한 클래스 ID

                        if cls in CLASS_MAPPING:  # 학습된 클래스인지 확인
                            detected_classes.add(CLASS_MAPPING[cls])
                            self.create_marker(box, i, CLASS_MAPPING[cls])

                    # 플래닝에서 사용할 객체 타입만 퍼블리시
                    for detected in detected_classes:
                        self.signal_publish(detected)

                self.visualization_publish.publish(markers)

                # 디버깅용 YOLO 출력 이미지
                cv2.imshow("YOLO Detection", plots)
                cv2.waitKey(1)

    def create_marker(self, box, i, class_name):
        """ 감지된 객체에 대한 2D 마커 생성 (하이퍼파라미터 높이 비율 사용) """
        center_2d_x = box
