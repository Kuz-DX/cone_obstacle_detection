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
        print(f"[System] Conda path added: {conda_site[0]}")

import numpy as np
import cv2
import torch
import argparse
from math import atan2, degrees, sqrt

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, Bool
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

# [핵심] CvBridge 사용 안 함 (NumPy 버전 충돌 원천 봉쇄)

# ==============================================================================
# 유틸리티 함수
# ==============================================================================
def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: return None
    try: return np.polyfit(points_y, points_x, order)
    except: return None

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size: cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return np.zeros_like(binary_mask)
    comps = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: comps.append((i, stats[i, cv2.CC_STAT_AREA]))
    comps.sort(key=lambda x: x[1], reverse=True)
    cleaned = np.zeros_like(binary_mask)
    for i in range(min(len(comps), 2)):
        idx = comps[i][0]
        cleaned[labels == idx] = 255
    return cleaned

def final_filter(bev_mask):
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=10000)
    f4 = keep_top2_components(f3, min_area=300)
    return f4

def overlay_polyline(image, coeff, color=(0, 0, 255), step=4, thickness=2):
    if coeff is None: return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w: draw_points.append((int(x), int(y)))
    if len(draw_points) > 1:
        cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, thickness)
    return image

# ==============================================================================
# ROS 2 노드 클래스
# ==============================================================================
class LaneFollowerNode(Node):
    def __init__(self, opt):
        super().__init__('lane_follower_node')
        self.opt = opt
        self.get_logger().info("[Lane Follower] Initializing...")

        # 1. 모델 로드
        device_str = self.opt.device
        if device_str.isdigit() and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_str}')
        else:
            self.device = torch.device('cpu')
        
        self.get_logger().info(f"Loading weights from: {opt.weights}")
        self.model = YOLO(opt.weights).to(self.device)

        # 2. BEV 파라미터 로드
        self.get_logger().info(f"Loading params from: {opt.param_file}")
        try:
            params = np.load(opt.param_file)
            self.bev_params = {'src_points': params['src_points'], 'dst_points': params['dst_points']}
            self.bev_w, self.bev_h = int(params['warp_w']), int(params['warp_h'])
            self.get_logger().info(f"[SUCCESS] Loaded Params: BEV Size={self.bev_w}x{self.bev_h}")
        except Exception as e:
            self.get_logger().error(f"[FATAL] Failed to load BEV params: {e}")
            self.get_logger().error(f"Check file path: {opt.param_file}")
            sys.exit(1)

        # 3. 주행 파라미터
        self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.0025, 1.25, 0.003578125 
        self.tracked_lanes = {'left': {'coeff': None, 'age': 0}, 'right': {'coeff': None, 'age': 0}}
        self.tracked_center_path = {'coeff': None}
        self.SMOOTHING_ALPHA = 0.6 
        self.MAX_LANE_AGE = 7 
        self.L = 0.73
        self.THROTTLE_MIN, self.THROTTLE_MAX = 0.4, 0.6
        self.MIN_LOOKAHEAD_DISTANCE, self.MAX_LOOKAHEAD_DISTANCE = 1.75, 2.35
        self.current_throttle = self.THROTTLE_MIN

        # 4. ROS Setup
        self.pub_steering = self.create_publisher(Float32, 'auto_steer_angle_lane', 1)
        self.pub_lane_status = self.create_publisher(Bool, 'lane_detection_status', 1)
        self.pub_path = self.create_publisher(Path, 'lane_path', 10)
        self.pub_drivable_area = self.create_publisher(CompressedImage, 'drivable_area', 10)
        
        # [통신] QoS 설정 적용 (카메라 연결 문제 해결)
        if 'compressed' in self.opt.topic:
            self.msg_type = CompressedImage
        else:
            self.msg_type = Image

        self.sub_image = self.create_subscription(
            self.msg_type, 
            self.opt.topic, 
            self.image_callback, 
            qos_profile_sensor_data
        )
        self.sub_throttle = self.create_subscription(Float32, 'auto_throttle', self.throttle_callback, 1)

        # [시각화] 창 크기 자동 조절 (여백 제거)
        cv2.namedWindow("Original Camera View", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Roboflow Detections (on BEV)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Final Path & Logs (on BEV)", cv2.WINDOW_AUTOSIZE)

        self.frame_count = 0

    def throttle_callback(self, msg):
        self.current_throttle = np.clip(msg.data, self.THROTTLE_MIN, self.THROTTLE_MAX)

    def do_bev_transform(self, image):
        # 파라미터 파일에 저장된 좌표 그대로 변환 (고해상도 대응)
        M = cv2.getPerspectiveTransform(self.bev_params['src_points'], self.bev_params['dst_points'])
        return cv2.warpPerspective(image, M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)

    def draw_lanes_on_original(self, image, left_coeff, right_coeff, center_coeff):
        # 역변환 행렬 (dst -> src)
        M_inv = cv2.getPerspectiveTransform(self.bev_params['dst_points'], self.bev_params['src_points'])
        draw_img = image.copy()

        # [수정] 주행 영역(Drivable Area) 시각화 (단일 차선 보정 포함)
        LANE_WIDTH_M = 1.5
        lane_width_pixels = LANE_WIDTH_M / self.m_per_pixel_x

        viz_left = left_coeff
        viz_right = right_coeff

        if left_coeff is not None and right_coeff is None:
            viz_right = left_coeff.copy()
            viz_right[-1] += lane_width_pixels
        elif right_coeff is not None and left_coeff is None:
            viz_left = right_coeff.copy()
            viz_left[-1] -= lane_width_pixels

        if viz_left is not None and viz_right is not None:
            ys = np.linspace(0, self.bev_h - 1, num=100)
            left_xs = np.polyval(viz_left, ys)
            right_xs = np.polyval(viz_right, ys)

            # BEV 좌표계에서 폴리곤 구성 (왼쪽 라인 -> 오른쪽 라인 역순)
            pts_left = np.stack([left_xs, ys], axis=1)
            pts_right = np.stack([right_xs, ys], axis=1)
            pts_bev = np.vstack([pts_left, pts_right[::-1]])

            # 원본 이미지 좌표계로 변환
            pts_bev = np.array([pts_bev], dtype=np.float32)
            pts_orig = cv2.perspectiveTransform(pts_bev, M_inv)

            # 반투명 오버레이 (초록색)
            overlay = draw_img.copy()
            cv2.fillPoly(overlay, [np.int32(pts_orig)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, draw_img, 0.7, 0, draw_img)

        def transform_and_draw(coeff, color):
            if coeff is None: return
            # BEV 상의 y좌표들 (0 ~ bev_h)
            ys = np.linspace(0, self.bev_h - 1, num=100)
            xs = np.polyval(coeff, ys)
            
            # (1, N, 2) 형태의 점들
            pts_bev = np.array([np.stack([xs, ys], axis=1)], dtype=np.float32)
            pts_orig = cv2.perspectiveTransform(pts_bev, M_inv)
            
            # pts_orig[0]는 (N, 2) 형태
            cv2.polylines(draw_img, [np.int32(pts_orig)[0]], False, color, 3)

        transform_and_draw(left_coeff, (255, 0, 0))   # Blue
        transform_and_draw(right_coeff, (0, 0, 255))  # Red
        transform_and_draw(center_coeff, (0, 255, 0)) # Green
        return draw_img

    def image_to_vehicle(self, pt_bev):
        u, v = pt_bev
        x_vehicle = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        y_vehicle = (self.bev_w / 2 - u) * self.m_per_pixel_x
        return x_vehicle, y_vehicle

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 60 == 0:
            if self.msg_type == Image:
                self.get_logger().info(f"[Alive] Image Size: {msg.width}x{msg.height}")
            else:
                self.get_logger().info(f"[Alive] Compressed Image Received")

        try:
            # [변환] CvBridge 대신 NumPy 수동 변환 (NumPy 2.x 충돌 방지)
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            
            if self.msg_type == Image:
                # YUYV (2 bytes/pixel)
                if np_arr.size == (msg.width * msg.height * 2): 
                    img = cv2.cvtColor(np_arr.reshape((msg.height, msg.width, 2)), cv2.COLOR_YUV2BGR_YUYV)
                
                # RGB8/BGR8 (3 bytes/pixel)
                elif np_arr.size == (msg.width * msg.height * 3):
                    img = np_arr.reshape((msg.height, msg.width, 3))
                    if 'rgb' in msg.encoding.lower(): 
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # CompressedImage
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None: return

            # [중요] Resize 없음! 원본 해상도 그대로 사용
            self.process_image(np.ascontiguousarray(img))

        except Exception as e:
            self.get_logger().error(f"Manual Decode Error: {e}")

    def process_image(self, im0s):
        # 변수 안전 초기화
        steer_deg = None
        goal_point_bev = None
        final_left_coeff = None
        final_right_coeff = None
        lane_detected_bool = False
        
        # 화면 초기화 (plot 실패 대비)
        annotated_frame = im0s.copy() 

        # 1. BEV Transform
        bev_image_input = self.do_bev_transform(im0s)
        
        # 2. Inference
        results = self.model(bev_image_input, imgsz=self.opt.img_size, conf=self.opt.conf_thres, 
                            iou=self.opt.iou_thres, device=self.device, verbose=False)
        result = results[0]

        # 3. Mask Processing
        combined_mask_bev = np.zeros(result.orig_shape[:2], dtype=np.uint8)
        if result.masks is not None:
            confidences = result.boxes.conf
            for i, mask_tensor in enumerate(result.masks.data):
                if confidences[i] >= self.opt.conf_thres:
                    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                    if mask_np.shape != result.orig_shape[:2]:
                        mask_np = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]))
                    combined_mask_bev = np.maximum(combined_mask_bev, mask_np)
        
        final_mask = final_filter(combined_mask_bev)
        bev_im_for_drawing = bev_image_input.copy()

        # 4. Lane Extraction
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        current_detections = []
        if num_labels > 1:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 100:
                    ys, xs = np.where(labels == i)
                    coeff = polyfit_lane(ys, xs, order=2)
                    if coeff is not None:
                        x_at_bottom = np.polyval(coeff, self.bev_h - 1)
                        current_detections.append({'coeff': coeff, 'x_bottom': x_at_bottom})
        current_detections.sort(key=lambda c: c['x_bottom'])

        # 5. Tracking
        left_lane_tracked = self.tracked_lanes['left']
        right_lane_tracked = self.tracked_lanes['right']
        current_left, current_right = None, None

        if len(current_detections) == 2:
            current_left, current_right = current_detections[0], current_detections[1]
        elif len(current_detections) == 1:
            detected_lane = current_detections[0]
            dist_to_left = abs(detected_lane['x_bottom'] - np.polyval(left_lane_tracked['coeff'], self.bev_h - 1)) if left_lane_tracked['coeff'] is not None else float('inf')
            dist_to_right = abs(detected_lane['x_bottom'] - np.polyval(right_lane_tracked['coeff'], self.bev_h - 1)) if right_lane_tracked['coeff'] is not None else float('inf')
            
            if dist_to_left < dist_to_right and left_lane_tracked['coeff'] is not None: current_left = detected_lane
            elif dist_to_right < dist_to_left and right_lane_tracked['coeff'] is not None: current_right = detected_lane
            else:
                if detected_lane['x_bottom'] < self.bev_w / 2: current_left = detected_lane
                else: current_right = detected_lane

        if current_left:
            if left_lane_tracked['coeff'] is None: left_lane_tracked['coeff'] = current_left['coeff']
            else: left_lane_tracked['coeff'] = (self.SMOOTHING_ALPHA * current_left['coeff'] + (1 - self.SMOOTHING_ALPHA) * left_lane_tracked['coeff'])
            left_lane_tracked['age'] = 0
        else:
            left_lane_tracked['age'] += 1
            if left_lane_tracked['age'] > self.MAX_LANE_AGE: left_lane_tracked['coeff'] = None

        if current_right:
            if right_lane_tracked['coeff'] is None: right_lane_tracked['coeff'] = current_right['coeff']
            else: right_lane_tracked['coeff'] = (self.SMOOTHING_ALPHA * current_right['coeff'] + (1 - self.SMOOTHING_ALPHA) * right_lane_tracked['coeff'])
            right_lane_tracked['age'] = 0
        else:
            right_lane_tracked['age'] += 1
            if right_lane_tracked['age'] > self.MAX_LANE_AGE: right_lane_tracked['coeff'] = None

        final_left_coeff, final_right_coeff = left_lane_tracked['coeff'], right_lane_tracked['coeff']
        lane_detected_bool = (final_left_coeff is not None) or (final_right_coeff is not None)
        self.pub_lane_status.publish(Bool(data=lane_detected_bool))

        # 6. Steering
        if lane_detected_bool:
            center_points = []
            LANE_WIDTH_M = 1.5
            lane_width_pixels = LANE_WIDTH_M / self.m_per_pixel_x
            
            for y in range(self.bev_h - 1, self.bev_h // 2, -1):
                x_center = None
                if final_left_coeff is not None and final_right_coeff is not None:
                    x_center = (np.polyval(final_left_coeff, y) + np.polyval(final_right_coeff, y)) / 2
                elif final_left_coeff is not None:
                    x_center = np.polyval(final_left_coeff, y) + lane_width_pixels / 2
                elif final_right_coeff is not None:
                    x_center = np.polyval(final_right_coeff, y) - lane_width_pixels / 2
                
                if x_center is not None: center_points.append([x_center, y])

            target_center_lane_coeff = None
            if len(center_points) > 10:
                target_center_lane_coeff = polyfit_lane(np.array(center_points)[:, 1], np.array(center_points)[:, 0], order=2)

            if target_center_lane_coeff is not None:
                if self.tracked_center_path['coeff'] is None: self.tracked_center_path['coeff'] = target_center_lane_coeff
                else: self.tracked_center_path['coeff'] = (self.SMOOTHING_ALPHA * target_center_lane_coeff + (1 - self.SMOOTHING_ALPHA) * self.tracked_center_path['coeff'])
            
            if self.tracked_center_path['coeff'] is not None:
                final_center_coeff = self.tracked_center_path['coeff']

                # [Path Publishing]
                path_msg = Path()
                path_msg.header.frame_id = "base_link"
                path_msg.header.stamp = self.get_clock().now().to_msg()
                
                for y_bev in range(self.bev_h - 1, 0, -20):
                    x_bev = np.polyval(final_center_coeff, y_bev)
                    x_veh, y_veh = self.image_to_vehicle((x_bev, y_bev))
                    
                    pose = PoseStamped()
                    pose.header = path_msg.header
                    pose.pose.position.x = float(x_veh)
                    pose.pose.position.y = float(y_veh)
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    path_msg.poses.append(pose)
                self.pub_path.publish(path_msg)

                throttle_range = self.THROTTLE_MAX - self.THROTTLE_MIN
                normalized_throttle = (self.current_throttle - self.THROTTLE_MIN) / throttle_range if throttle_range > 0 else 0.0
                dynamic_lookahead_distance = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE) * normalized_throttle
                
                for y_bev in range(self.bev_h - 1, -1, -1):
                    x_bev = np.polyval(final_center_coeff, y_bev)
                    x_veh, y_veh_right = self.image_to_vehicle((x_bev, y_bev))
                    dist = sqrt(x_veh**2 + y_veh_right**2)
                    if dist >= dynamic_lookahead_distance:
                        goal_point_bev = (int(x_bev), int(y_bev))
                        steer_rad = atan2(2.0 * self.L * y_veh_right, x_veh**2 + y_veh_right**2)
                        steer_deg = np.clip(-degrees(steer_rad), -25.0, 25.0)
                        self.pub_steering.publish(Float32(data=steer_deg))
                        break

        # 7. Visualization
        try:
            annotated_frame = result.plot()
        except:
            pass 
        
        overlay_polyline(bev_im_for_drawing, final_left_coeff, color=(255, 0, 0), step=2, thickness=2)
        overlay_polyline(bev_im_for_drawing, final_right_coeff, color=(0, 0, 255), step=2, thickness=2)
        if self.tracked_center_path['coeff'] is not None:
            overlay_polyline(bev_im_for_drawing, self.tracked_center_path['coeff'], color=(0, 255, 0), step=2, thickness=3)

        if goal_point_bev is not None:
            cv2.circle(bev_im_for_drawing, goal_point_bev, 10, (0, 255, 255), -1)

        steer_text = f"Steer: {steer_deg:.1f} deg" if steer_deg is not None else "Steer: N/A"
        cv2.putText(bev_im_for_drawing, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(bev_im_for_drawing, f"Lane Detected: {lane_detected_bool}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        throttle_range = self.THROTTLE_MAX - self.THROTTLE_MIN
        norm_thr = (self.current_throttle - self.THROTTLE_MIN) / throttle_range if throttle_range > 0 else 0.0
        viz_lookahead = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE) * norm_thr
        cv2.putText(bev_im_for_drawing, f"Lookahead: {viz_lookahead:.2f}m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(bev_im_for_drawing, f"Throttle: {self.current_throttle:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 원본 화면에 역변환된 차선 표시
        original_with_lanes = self.draw_lanes_on_original(im0s, final_left_coeff, final_right_coeff, self.tracked_center_path['coeff'])
        
        # [Publish] 주행 가능 영역 이미지 발행
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.format = "jpeg"
        success, encoded_img = cv2.imencode('.jpg', original_with_lanes)
        if success:
            msg.data = encoded_img.tobytes()
            self.pub_drivable_area.publish(msg)

        cv2.imshow("Original Camera View", original_with_lanes)
        cv2.imshow("Roboflow Detections (on BEV)", annotated_frame)
        cv2.imshow("Final Path & Logs (on BEV)", bev_im_for_drawing)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    
    # [수정] ROS2 패키지 공유 디렉토리에서 파일 경로를 동적으로 찾도록 수정
    package_share_directory = get_package_share_directory('yolotl_ros2')
    default_weights = os.path.join(package_share_directory, 'config', 'weights3.pt')
    default_params = os.path.join(package_share_directory, 'config', 'bev_params_7.npz')

    # 사용자가 직접 경로를 지정할 수 있도록 옵션은 유지
    parser.add_argument('--weights', default=default_weights, help='Path to model weights')
    parser.add_argument('--param-file', default=default_params, help='Path to BEV parameters file')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.6)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--device', default='0')
    parser.add_argument('--topic', type=str, default='/image_raw/compressed', help='ROS 2 Image Topic')
    opt, _ = parser.parse_known_args()
    
    node = LaneFollowerNode(opt)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()