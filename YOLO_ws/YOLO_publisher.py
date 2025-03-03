#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
import torch
from ultralytics import YOLO
from sensor_msgs.msg import LaserScan, CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import time
from scipy.spatial import KDTree
from math import cos, sin

# YOLO 학습된 클래스
CON = 0  # 라바콘
DRUM = 1  # PE드럼

# 하이퍼파라미터 : LIDAR가 bbox 높이의 몇 % 높이에 있는지 (라바콘 기준으로 설정하면 될듯)
HEIGHT_RATIO = 0.3  # 0.0 ~ 1.0

class ConeDrumDetection:
    def __init__(self):
        rospy.init_node('cone_drum_detector', anonymous=True)

        # ROS 파라미터에서 높이 비율을 조정 가능하도록 설정
        self.height_ratio = rospy.get_param("~height_ratio", HEIGHT_RATIO)

        # 퍼블리셔
        self.visualization_publish = rospy.Publisher("/yolo_viz", MarkerArray, queue_size=10)
        self.image_pub = rospy.Publisher("/yolo_debug", CompressedImage, queue_size=10)  # SSH에서도 이미지 볼 수 있게

        # 섭스크라이버  ############ 이미지 토픽 확인하기
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback, queue_size=10)
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)

        # YOLO 모델 로드
        self.model = YOLO('/home/user/YOLO_ws/weights/YOLO_0216.pt') ############ 경로 수정하기

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
                res = self.model.predict(self.img_bgr, stream=True)

                markers = MarkerArray()

                if len(res[0].boxes) > 0:
                    # LiDAR 데이터 예외 처리
                    if self.filtered_points is None or len(self.filtered_points) == 0:
                        rospy.logwarn("No valid LiDAR points available.")
                        return  # LiDAR 데이터가 없으면 탐지 수행 X

                    for i, box in enumerate(res[0].boxes):
                        cls = int(box.cls.item())  # YOLO가 예측한 클래스 ID

                        if cls == CON:
                            self.create_marker(box, i, "con", (1.0, 0.5, 0.0))
                        elif cls == DRUM:
                            self.create_marker(box, i, "drum", (1.0, 0.0, 0.0))

                self.visualization_publish.publish(markers)

                # SSH 사용 시 토픽으로 YOLO 탐지 결과 퍼블리시
                msg = self.bridge.cv2_to_compressed_imgmsg(res[0].plot())
                self.image_pub.publish(msg)

    def create_marker(self, box, i, label, color):
        """ 감지된 객체에 대한 2D 마커 생성 """
        center_2d_x = box.xywh[0, 0].item()
        center_2d_y = box.xywh[0, 1].item()
        box_height = box.xywh[0, 3].item()

        # 하이퍼파라미터 적용
        target_y = center_2d_y - (box_height * self.height_ratio)

        # LiDAR 데이터 예외 처리
        if self.filtered_points is None or len(self.filtered_points) == 0:
            rospy.logwarn("No valid LiDAR points available for marker.")
            return

        # KNN (YOLO 중심점과 가장 가까운 LiDAR 점 찾기)
        kdtree = KDTree(self.filtered_points)
        _, index = kdtree.query([center_2d_x, target_y])
        closest_point_2d = self.filtered_points[index]

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.ns = label
        marker.id = i
        marker.lifetime = rospy.Duration(3.0)

        marker.pose.position.x = closest_point_2d[0]
        marker.pose.position.y = closest_point_2d[1]
        marker.pose.position.z = 0.0  

        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5 if label == "con" else 0.7

        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 0.5

        self.visualization_publish.publish(marker)

if __name__ == '__main__':
    detector = ConeDrumDetection()
    rospy.spin()
