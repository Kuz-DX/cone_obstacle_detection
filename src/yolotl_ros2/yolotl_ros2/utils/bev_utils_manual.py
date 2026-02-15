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

class BEVManualSetupNode(Node):
    def __init__(self, args):
        super().__init__('bev_manual_setup_node')
        self.args = args
        self.src_points = []
        self.max_points = 4
        
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

        print("\n[사용 방법 - Manual Mode (ROS 2 Topic)]")
        print("1. 클릭 순서: 좌하 -> 우하 -> 좌상 -> 우상")
        print("2. 'r' 키: 좌표 리셋")
        print("3. 's' 키: 파라미터 저장 및 종료")
        print("4. 'q' 키: 저장하지 않고 종료\n")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.src_points) < self.max_points:
                self.src_points.append((x, y))
                point_order = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
                current_point_index = len(self.src_points) - 1
                print(f"[INFO] Added {point_order[current_point_index]} point: ({x}, {y}) ({len(self.src_points)}/{self.max_points})")

                if len(self.src_points) == self.max_points:
                    print("[INFO] 모든 점이 선택되었습니다. 's'를 눌러 저장하거나 'r'로 리셋하세요.")
            else:
                print("[WARNING] 이미 4개의 점이 선택되었습니다. 리셋하려면 'r'을 누르세요.")

    def image_callback(self, msg):
        try:
            # Manual Decode
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

            disp = frame.copy()
            point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]
            
            for i, pt in enumerate(self.src_points):
                cv2.circle(disp, pt, 5, (0, 255, 0), -1)
                label = point_labels[i] if i < len(point_labels) else f"{i+1}"
                cv2.putText(disp, label, (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
            elif key == ord('s'):
                if len(self.src_points) < 4:
                    print("[WARNING] 4개의 점을 모두 선택해야 저장 가능합니다.")
                else:
                    np.savez(self.args.out_npz, src_points=np.float32(self.src_points), dst_points=self.dst_points, warp_w=self.args.warp_width, warp_h=self.args.warp_height)
                    try:
                        with open(self.args.out_txt, 'w') as f:
                            f.write("# Selected BEV Points\n")
                            for i, pt in enumerate(self.src_points):
                                f.write(f"{pt[0]}, {pt[1]} # {point_labels[i]}\n")
                        print(f"[INFO] 저장 완료: {self.args.out_npz}, {self.args.out_txt}")
                    except Exception as e:
                        print(f"[ERROR] 파일 저장 중 오류 발생: {e}")
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Image Callback Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='/image_raw/compressed', help='구독할 ROS 2 이미지 토픽 이름')
    parser.add_argument('--warp-width', type=int, default=640)
    parser.add_argument('--warp-height', type=int, default=640)
    parser.add_argument('--out-npz', type=str, default='bev_params_manual.npz')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_manual.txt')
    args = parser.parse_args()
    
    node = BEVManualSetupNode(args)
    
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