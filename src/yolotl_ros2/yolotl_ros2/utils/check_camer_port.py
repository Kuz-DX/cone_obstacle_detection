#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Script: Check Available Camera Ports
-------------------------------------------
ROS 2 호환 버전: 시스템에서 사용 가능한 비디오 장치(/dev/videoX) 번호를 탐색합니다.
"""

import cv2

def find_available_cameras(max_index=10):
    """
    0번부터 max_index까지의 포트를 검색하여 사용 가능한 인덱스 리스트를 반환합니다.
    """
    available_ports = []
    for i in range(max_index):
        # ROS 2(Linux) 환경에서는 CAP_V4L2를 명시하는 것이 인덱싱에 더 정확합니다.
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            available_ports.append(i)
            cap.release()
    return available_ports

def main():
    print("[INFO] 사용 가능한 카메라 포트를 검색 중입니다...")
    ports = find_available_cameras()
    
    if ports:
        print(f"✅ 사용 가능한 카메라 포트 리스트: {ports}")
        print(f"가장 권장되는 포트 번호는 {ports[0]}번입니다.")
    else:
        print("❌ 사용 가능한 카메라를 찾을 수 없습니다. 연결 상태를 확인하세요.")

if __name__ == "__main__":
    main()