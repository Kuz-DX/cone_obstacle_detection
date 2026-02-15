#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import glob

# [1] Conda 환경 라이브러리 경로 우선순위 설정 (NumPy 충돌 방지)
if 'CONDA_PREFIX' in os.environ:
    conda_site = glob.glob(os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'python*', 'site-packages'))
    if conda_site: 
        sys.path.insert(0, conda_site[0])

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
import cv2
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection_node')
        
        # ROS 2 파라미터 선언 (기본값 설정)
        self.declare_parameter('weights', 'cone.pt')
        self.declare_parameter('topic', '/image_raw/compressed')
        self.declare_parameter('conf', 0.3)
        self.declare_parameter('device', '')

        # 파라미터 값 읽기
        weights_filename = self.get_parameter('weights').value
        self.topic_name = self.get_parameter('topic').value
        self.conf_thres = self.get_parameter('conf').value
        self.device = self.get_parameter('device').value

        # 모델 파일 경로 찾기
        # 1. 패키지 share 디렉토리 확인
        try:
            pkg_share = get_package_share_directory('dynamic_cone')
            weights_path = os.path.join(pkg_share, 'config', weights_filename)
        except:
            weights_path = os.path.join('config', weights_filename)

        # 2. 로컬 경로 확인 (개발 환경)
        if not os.path.exists(weights_path):
            local_path = os.path.join(os.getcwd(), 'config', weights_filename)
            if os.path.exists(local_path):
                weights_path = local_path
            elif os.path.exists(weights_filename):
                weights_path = weights_filename
            else:
                self.get_logger().warn(f"Weights file not found: {weights_path}. Using raw string.")
                weights_path = weights_filename

        self.get_logger().info(f"Loading YOLO model from: {weights_path}")
        
        # YOLO 모델 로드
        try:
            self.model = YOLO(weights_path)
            if self.device:
                self.model.to(self.device)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

        # 이미지 Subscriber 설정
        if 'compressed' in self.topic_name:
            self.sub = self.create_subscription(
                CompressedImage, 
                self.topic_name, 
                self.image_callback, 
                qos_profile_sensor_data
            )
            self.is_compressed = True
        else:
            self.sub = self.create_subscription(
                Image, 
                self.topic_name, 
                self.image_callback, 
                qos_profile_sensor_data
            )
            self.is_compressed = False

        # 시각화 창 생성
        cv2.namedWindow("Cone Detection", cv2.WINDOW_AUTOSIZE)
        self.get_logger().info("Cone Detection Node Initialized.")

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            
            # 이미지 디코딩 (CvBridge 미사용 - NumPy 충돌 방지)
            if self.is_compressed:
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                if np_arr.size == (msg.width * msg.height * 2): # YUYV
                    img = cv2.cvtColor(np_arr.reshape((msg.height, msg.width, 2)), cv2.COLOR_YUV2BGR_YUYV)
                elif np_arr.size == (msg.width * msg.height * 3): # RGB/BGR
                    img = np_arr.reshape((msg.height, msg.width, 3))
                    if 'rgb' in msg.encoding.lower(): 
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None: return

            # YOLO 추론
            results = self.model(img, conf=self.conf_thres, verbose=False)
            
            # 결과 시각화 (Bounding Box 그리기)
            annotated_frame = results[0].plot()

            # 화면 출력
            cv2.imshow("Cone Detection", annotated_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
