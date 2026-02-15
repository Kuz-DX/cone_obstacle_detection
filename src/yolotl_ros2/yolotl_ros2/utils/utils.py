#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
import re
import glob
import random
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# --- [기존 유틸리티 함수들 유지] ---

def git_describe(path=Path(__file__).parent):
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError:
        return ''

def date_modified(path=__file__):
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def select_device(device='', batch_size=None):
    # YOLOPv2 로깅 정보 출력
    s = f'YOLOPv2 🚀 {git_describe() or date_modified()} torch {torch.__version__} '
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            s += f"CUDA:{i} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    else:
        s += 'CPU\n'
    print(s)
    return torch.device('cuda:0' if cuda else 'cpu')

def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# --- [이미지 전처리 및 후처리 함수] ---

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    # YOLOv5/v2 스타일의 NMS 로직 (기존 코드 유지)
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]: continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]: continue
        i = torchvision.ops.nms(x[:, :4], x[:, 4], iou_thres)
        output[xi] = x[i]
    return output

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def driving_area_mask(seg):
    # 주행 영역 마스크 생성
    da_predict = seg[:, :, 12:372, :]
    da_seg_mask = F.interpolate(da_predict, scale_factor=2, mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    return da_seg_mask.int().squeeze().cpu().numpy()

def lane_line_mask(ll, threshold=0.5, method='otsu'):
    # 차선 마스크 생성 (기존 로직 유지)
    ll_predict = ll[:, :, 12:372, :]
    ll_seg_map = F.interpolate(ll_predict, scale_factor=2, mode='bilinear').squeeze(1)[0].cpu().numpy()
    ll_seg_map_8u = (ll_seg_map * 255).astype(np.uint8)
    guided = cv2.ximgproc.guidedFilter(guide=ll_seg_map_8u, src=ll_seg_map_8u, radius=4, eps=1e-1)
    if method == 'otsu':
        _, binary_mask = cv2.threshold(guided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary_mask = (guided > threshold * 255).astype(np.uint8) * 255
    return binary_mask

# --- [ROS 2 이미지 입력을 위한 변환 유틸리티] ---

def prepare_input_from_ros(img0, img_size=640, stride=32):
    """
    ROS 2 토픽으로 받은 이미지를 모델 입력 규격으로 변환
    """
    img0 = cv2.resize(img0, (1280, 720)) # 강제 리사이즈 (기존 요구사항)
    img = letterbox(img0, (img_size, img_size), stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0

# --- [기타 클래스들 유지] ---
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
    # ... (기존 메트릭 계산 로직 그대로 사용 가능)