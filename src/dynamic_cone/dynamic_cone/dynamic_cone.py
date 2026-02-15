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
from nav_msgs.msg import Path
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
        self.declare_parameter('bev_params', 'bev_params_7.npz')
        self.declare_parameter('obstacle_threshold', 0.1)

        # 파라미터 값 읽기
        weights_filename = self.get_parameter('weights').value
        self.topic_name = self.get_parameter('topic').value
        self.conf_thres = self.get_parameter('conf').value
        self.device = self.get_parameter('device').value
        bev_filename = self.get_parameter('bev_params').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value

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

        # BEV 파라미터 로드
        self.M_inv = None
        try:
            # yolotl_ros2 패키지에서 우선 검색
            try:
                yolo_share = get_package_share_directory('yolotl_ros2')
                bev_path = os.path.join(yolo_share, 'config', bev_filename)
            except:
                bev_path = bev_filename

            if not os.path.exists(bev_path):
                bev_path = os.path.join(os.getcwd(), 'config', bev_filename)

            if os.path.exists(bev_path):
                params = np.load(bev_path)
                self.bev_w, self.bev_h = int(params['warp_w']), int(params['warp_h'])
                M = cv2.getPerspectiveTransform(params['src_points'], params['dst_points'])
                self.M_inv = np.linalg.inv(M)
                self.get_logger().info(f"Loaded BEV params from: {bev_path}")
            else:
                self.get_logger().warn(f"BEV params file not found: {bev_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load BEV params: {e}")

        # 좌표 변환 상수 (demo_with_ros2.py와 동일)
        self.m_per_pixel_y = 0.0025
        self.y_offset_m = 1.25
        self.m_per_pixel_x = 0.003578125

        self.get_logger().info(f"Loading YOLO model from: {weights_path}")
        
        # YOLO 모델 로드
        try:
            self.model = YOLO(weights_path)
            if self.device:
                self.model.to(self.device)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

        # Path 구독
        self.latest_path = None
        self.sub_path = self.create_subscription(Path, '/lane_path', self.path_callback, 10)

        # Drivable Area 구독
        self.latest_drivable_img = None
        self.sub_drivable = self.create_subscription(CompressedImage, 'drivable_area', self.drivable_callback, 10)

        # 결과 이미지 퍼블리셔
        self.pub_result = self.create_publisher(CompressedImage, '/cone_detection/compressed', 10)

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

    def path_callback(self, msg):
        self.latest_path = msg

    def drivable_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if decoded_img is not None:
                self.latest_drivable_img = decoded_img
        except Exception as e:
            self.get_logger().error(f"Drivable callback error: {e}")

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
            # 배경 이미지 설정
            if self.latest_drivable_img is not None:
                bg_img = self.latest_drivable_img.copy()
                if bg_img.shape[:2] != img.shape[:2]:
                    bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))
            else:
                bg_img = img.copy()
            
            annotated_frame = bg_img.copy()

            # [추가] 주행 가능 영역 마스크 생성 (초록색 영역 감지)
            drivable_mask = None
            if self.latest_drivable_img is not None:
                hsv = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2HSV)
                # 초록색 범위 (demo_with_ros2.py의 오버레이 감지)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                drivable_mask = cv2.inRange(hsv, lower_green, upper_green)

            # [수정] 바운딩 박스 수동 그리기 (장애물 판단 로직 포함)
            obstacle_count = 0
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"
                color = (0, 165, 255) # 기본: 주황색

                # 장애물 판단
                if drivable_mask is not None:
                    bx1, by1 = max(0, x1), max(0, y1)
                    bx2, by2 = min(drivable_mask.shape[1], x2), min(drivable_mask.shape[0], y2)
                    if bx2 > bx1 and by2 > by1:
                        roi = drivable_mask[by1:by2, bx1:bx2]
                        overlap_ratio = cv2.countNonZero(roi) / ((bx2 - bx1) * (by2 - by1))
                        if overlap_ratio > self.obstacle_threshold:
                            color = (0, 0, 255) # 장애물: 빨간색
                            label += " [OBS]"
                            obstacle_count += 1

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w_t, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # [추가] 장애물(콘)이 2개 이상이면 EMERGENCY 표시 (크고 굵게)
            if obstacle_count >= 2:
                cv2.putText(annotated_frame, "EMERGENCY", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)

            # Lane Path 오버레이
            if self.latest_path is not None and self.M_inv is not None:
                path_points_bev = []
                for pose in self.latest_path.poses:
                    x_veh = pose.pose.position.x
                    y_veh = pose.pose.position.y
                    
                    # Vehicle(m) -> BEV(pixel) 변환
                    v = self.bev_h - (x_veh - self.y_offset_m) / self.m_per_pixel_y
                    u = self.bev_w / 2 - y_veh / self.m_per_pixel_x
                    path_points_bev.append([u, v])
                
                if path_points_bev:
                    # BEV(pixel) -> Original(pixel) 역변환
                    pts_bev = np.array([path_points_bev], dtype=np.float32) # (1, N, 2)
                    pts_orig = cv2.perspectiveTransform(pts_bev, self.M_inv)
                    cv2.polylines(annotated_frame, [np.int32(pts_orig)], False, (0, 255, 0), 3)

            # [Publish] 결과 이미지 발행 (CompressedImage)
            msg_pub = CompressedImage()
            msg_pub.header.stamp = self.get_clock().now().to_msg()
            msg_pub.header.frame_id = "base_link"
            msg_pub.format = "jpeg"
            success, encoded_img = cv2.imencode('.jpg', annotated_frame)
            if success:
                msg_pub.data = encoded_img.tobytes()
                self.pub_result.publish(msg_pub)

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
