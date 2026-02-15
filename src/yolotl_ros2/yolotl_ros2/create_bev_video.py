#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
from ament_index_python.packages import get_package_share_directory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='output_bev.mp4', help='Output video path')
    parser.add_argument('--margin-x', type=int, default=0, help='Offset for BEV view horizontal')
    parser.add_argument('--margin-y', type=int, default=0, help='Offset for BEV view vertical')
    args = parser.parse_args()

    # 파라미터 로드
    try:
        pkg_share = get_package_share_directory('yolotl_ros2')
        param_path = os.path.join(pkg_share, 'config', 'bev_params_y_5.npz')
    except:
        param_path = './config/bev_params_y_5.npz'

    if not os.path.exists(param_path):
        print(f"Error: Param file not found at {param_path}")
        return

    params = np.load(param_path)
    
    # 원본 dst_points 로드
    dst_points = params['dst_points']

    # 마진 적용
    margin_x = args.margin_x
    margin_y = args.margin_y
    
    dst_points_with_margin = np.float32([
        [0 + margin_x, params['warp_h'] - margin_y], # 좌하
        [params['warp_w'] - margin_x, params['warp_h'] - margin_y], # 우하
        [0 + margin_x, 0 + margin_y], # 좌상
        [params['warp_w'] - margin_x, 0 + margin_y]  # 우상
    ])
    
    M = cv2.getPerspectiveTransform(params['src_points'], dst_points_with_margin)
    h, w = int(params['warp_h']), int(params['warp_w'])

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, (w, h))

    print(f"Processing {args.source} -> {args.output} ...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        bev_frame = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
        out.write(bev_frame)
        cv2.imshow("BEV Preview", bev_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    main()