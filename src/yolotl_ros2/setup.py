from setuptools import setup
import os
from glob import glob

package_name = 'yolotl_ros2'

setup(
    name=package_name,
    version='0.0.0',
    # 패키지 소스 폴더와 utils 서브 모듈 포함
    packages=[package_name, package_name + '.utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # config 폴더 안의 모델 가중치(.pt)와 파라미터(.npz)를 설치 경로로 복사
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='j',
    maintainer_email='seonjuhan1@gmail.com',
    description='YOLO based Lane Segmentation for ROS 2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 1. 메인 자율주행 노드 (demo_with_ros2.py)
            'lane_follower = yolotl_ros2.demo_with_ros2:main',
            
            # 2. 로컬 테스트 노드 (demo.py)
            'local_demo = yolotl_ros2.demo:main',
            
            # 3. BEV 변환 영상 생성 (create_bev_video.py)
            'bev_video_creator = yolotl_ros2.create_bev_video:main',
            
            # 4. 실시간 듀얼 레코더 (create_realtime_bev_video_dual_record.py)
            'dual_recorder = yolotl_ros2.create_realtime_bev_video_dual_record:main',
            
            # 5. 기본 결과 노드 (basic_result.py)
            'basic_follower = yolotl_ros2.basic_result:ros_main',
        ],
    },
)