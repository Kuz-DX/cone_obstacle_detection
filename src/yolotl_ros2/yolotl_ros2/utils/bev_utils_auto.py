#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
import argparse
import sys

class BEVAutoSetupNode(Node):
    def __init__(self, args):
        super().__init__('bev_auto_setup_node')
        self.args = args
        self.src_points = []
        self.max_points = 4
        self.current_frame_width = 0
        
        self.dst_points = np.float32([
            [0, args.warp_height],
            [args.warp_width, args.warp_height],
            [0, 0],
            [args.warp_width, 0]
        ])

        if 'compressed' in args.topic:
            self.msg_type = CompressedImage
        else:
            self.msg_type = Image

        self.subscription = self.create_subscription(
            self.msg_type,
            args.topic,
            self.image_callback,
            qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribing to {args.topic}...")

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Original", self.mouse_callback)

        print("\n[사용 방법 - Auto Mode (ROS 2 Topic)]")
        print("1. 클릭 순서: 좌하단 -> 좌상단 (우측 점들은 대칭으로 자동 생성)")
        print("2. 's' 키: 저장 후 종료 / 'r' 키: 리셋 / 'q' 키: 취소\n")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_frame_width == 0:
                self.get_logger().warn("프레임 너비가 설정되지 않았습니다.")
                return

            if len(self.src_points) < self.max_points:
                if len(self.src_points) == 0: # 첫 번째 클릭 (좌하단)
                    self.src_points.append((x, y))
                    symmetric_x = self.current_frame_width - 1 - x
                    self.src_points.append((symmetric_x, y))
                    print(f"[INFO] 좌하단 추가: ({x}, {y}), 우하단 자동추가: ({symmetric_x}, {y})")
                elif len(self.src_points) == 2: # 세 번째 클릭 (좌상단)
                    self.src_points.append((x, y))
                    symmetric_x = self.current_frame_width - 1 - x
                    self.src_points.append((symmetric_x, y))
                    print(f"[INFO] 좌상단 추가: ({x}, {y}), 우상단 자동추가: ({symmetric_x}, {y})")
                    print("[INFO] 4개 점 선택 완료. 's'를 눌러 저장하거나 'r'로 초기화하세요.")
            else:
                print("[WARNING] 이미 4개의 점이 선택되었습니다.")

    def image_callback(self, msg):
        try:
            # ROS Image -> OpenCV Image (Manual Decode)
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            
            if self.msg_type == Image:
                if np_arr.size == (msg.width * msg.height * 2): 
                    frame = cv2.cvtColor(np_arr.reshape((msg.height, msg.width, 2)), cv2.COLOR_YUV2BGR_YUYV)
                elif np_arr.size == (msg.width * msg.height * 3):
                    frame = np_arr.reshape((msg.height, msg.width, 3))
                    if 'rgb' in msg.encoding.lower():
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None: return

            self.current_frame_width = frame.shape[1]
            disp = frame.copy()

            for i, pt in enumerate(self.src_points):
                cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            
            if len(self.src_points) == 4:
                cv2.polylines(disp, [np.array(self.src_points, dtype=np.int32)], True, (0,0,255), 2)
                M = cv2.getPerspectiveTransform(np.float32(self.src_points), self.dst_points)
                bev_result = cv2.warpPerspective(frame, M, (self.args.warp_width, self.args.warp_height))
                cv2.imshow("BEV", bev_result)

            cv2.imshow("Original", disp)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                rclpy.shutdown()
            elif key == ord('r'):
                self.src_points = []
            elif key == ord('s') and len(self.src_points) == 4:
                np.savez(self.args.out_npz, src_points=np.float32(self.src_points), dst_points=self.dst_points, warp_w=self.args.warp_width, warp_h=self.args.warp_height)
                with open(self.args.out_txt, 'w') as f:
                    for pt in self.src_points: f.write(f"{pt[0]}, {pt[1]}\n")
                print(f"[INFO] 저장 완료: {self.args.out_npz}")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Image Callback Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='/image_raw/compressed', help='구독할 ROS 2 이미지 토픽 이름')
    parser.add_argument('--warp-width', type=int, default=640)
    parser.add_argument('--warp-height', type=int, default=640)
    parser.add_argument('--out-npz', type=str, default='bev_params_3.npz')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_3.txt')
    args = parser.parse_args()
    
    node = BEVAutoSetupNode(args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()