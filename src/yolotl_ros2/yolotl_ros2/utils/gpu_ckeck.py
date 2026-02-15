#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Script: GPU Status Check
--------------------------------
ROS 2 호환 버전: PyTorch를 사용하여 CUDA 가속 가능 여부 및 GPU 정보를 확인합니다.
"""

import torch

def check_gpu_status():
    """
    GPU 장치의 이름과 사용 가능한 장치 개수를 반환하고 출력합니다.
    """
    is_available = torch.cuda.is_available()
    
    if not is_available:
        print("❌ [WARNING] CUDA를 사용할 수 없습니다. CPU 모드로 동작합니다.")
        return False, None, 0

    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    
    print(f"✅ [INFO] GPU 사용 가능 확인")
    print(f"   - 장치 이름: {device_name}")
    print(f"   - 가용한 GPU 개수: {device_count}")
    
    return True, device_name, device_count

def main():
    check_gpu_status()

if __name__ == "__main__":
    main()