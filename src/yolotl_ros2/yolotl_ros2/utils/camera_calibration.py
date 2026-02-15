#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Script: Camera Calibration (Checkerboard)
---------------------------------------------------
ROS 2 호환 버전: 지정된 디렉토리의 이미지를 읽어 카메라 행렬과 왜곡 계수를 계산합니다.
"""

import cv2
import numpy as np
import os
import glob
import argparse

def run_calibration(image_dir, board_w=7, board_h=10, save_path='camera_params.npz'):
    # 체커보드 설정
    CHECKERBOARD = (board_w, board_h)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D 실제 세계 좌표 및 2D 이미지 좌표 저장용 리스트
    objpoints = []
    imgpoints = []

    # 실제 세계 좌표 초기화 (0,0,0), (1,0,0), ..., (6,9,0)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # 이미지 경로 탐색
    image_pattern = os.path.join(image_dir, '*.jpg')
    images = glob.glob(image_pattern)

    if not images:
        print(f"[ERROR] '{image_dir}' 경로에서 이미지를 찾을 수 없습니다.")
        return

    print(f"[INFO] 총 {len(images)}개의 이미지를 분석합니다...")

    gray = None
    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 코너 찾기
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            # 코너 좌표 정교화
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # 시각화 (선택 사항)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Calibration - Press any key', img)
            cv2.waitKey(100) # 0.1초씩 보여주며 진행

    cv2.destroyAllWindows()

    if len(objpoints) > 0:
        print("[INFO] 캘리브레이션 계산 중...")
        # 캘리브레이션 수행
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        print("\n=== Camera Calibration Result ===")
        print(f"Intrinsic Matrix (mtx):\n{mtx}")
        print(f"Distortion Coefficients (dist):\n{dist}")

        # 결과 저장 (NPZ 파일)
        np.savez(save_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print(f"\n[INFO] 결과가 '{save_path}'에 저장되었습니다.")
    else:
        print("[ERROR] 코너를 검출한 이미지가 없습니다. 캘리브레이션에 실패했습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 사용자 환경에 맞게 경로를 인자로 받도록 수정
    parser.add_argument('--image-dir', type=str, default='/home/highsky/Pictures/Webcam/',
                        help='체커보드 이미지들이 저장된 디렉토리 경로')
    parser.add_argument('--board-w', type=int, default=7, help='체커보드 가로 코너 수')
    parser.add_argument('--board-h', type=int, default=10, help='체커보드 세로 코너 수')
    parser.add_argument('--output', type=str, default='camera_params.npz', help='결과 저장 파일명')
    
    args = parser.parse_args()
    
    run_calibration(args.image_dir, args.board_w, args.board_h, args.output)