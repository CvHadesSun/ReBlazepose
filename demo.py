'''A demo to test the pose estimation using BlazePose landmarks detection model
@author:cvhades
date:2021-7-6-11:25'''
import os
import cv2
import sys
import numpy as np

from config.cfg import cfg
from pose_estimation.model import BlazePose
from pose_detection.model import PoseDetection
from utils import tools
from utils import img_process_tools

root_path = os.path.abspath(os.path.join('.'))


def singleImageDemo(image):
    # 1.update the cfg
    cfg['root_dir'] = root_path
    # 2.init the model
    pose_model = BlazePose(cfg)
    detection_model = PoseDetection(cfg)

    norm_roi = detection_model(image)  # normalized roi
    # adj_detection_roi=img_process_tools.TransformNormalizedRect(norm_roi,image.shape[1],image.shape[0])
    coord, pose_roi = pose_model(img=image, roi=norm_roi)

    # c2, pose_roi = pose_model(img=image, roi=pose_roi)

    vis_img = tools.draw2DJoint(image, coord[:33, :2])
    cv2.imshow('singleImageDemo', vis_img)
    cv2.waitKey()


def webCamDemo():
    # 1.update the cfg
    cfg['root_dir'] = root_path
    # 2.init the model
    pose_model = BlazePose(cfg)
    detection_model = PoseDetection(cfg)

    # cap = cv2.VideoCapture('./data/1_bodyweight_squats__tc__.webm')
    cap = cv2.VideoCapture('./data/1623983605589_video_camera_CurtsyLungetoBalanceRight.mp4')
    # cap = cv2.VideoCapture('./data/1623983453547_video_camera_PushUps.mp4')
    window_name = 'blazepose'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # first_frame = True
    ret, frame = cap.read()

    if ret:
        detection_roi = detection_model(frame)
        if detection_roi.width <= 0.001 or detection_roi.height <= 0.001:
            detection_roi = img_process_tools.ROI(0.5, 0.5, 1.0, 1.0, 0)
        # coord, pose_roi = pose_model(img=frame, roi=detection_roi)

    pre_pose_roi = detection_roi
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            # if first_frame:
            # detection_roi = detection_model(frame)
            # if detection_roi.width<=0.001 or detection_roi.height<=0.001:
            detection_roi = img_process_tools.ROI(0.5, 0.5, 1.0, 1.0, 0)

            coord, pose_roi = pose_model(img=frame, roi=detection_roi)
            # i+=1
            # if i==10:
            #
            # pre_pose_roi = pose_roi
            #     i=0
            # if pose_roi.width<=0.001 or pose_roi.height<=0.001:
            #     pose_roi=img_process_tools.ROI(0.5, 0.5, 1.0, 1.0, 0)
            # first_frame = False
            # else:
            #     coord, pose_roi = pose_model(img=frame, roi=pose_roi)
            # tools.plot_detections(frame,detection_model.final_detection[0])

            vis_img = tools.draw2DJoint(frame, coord[:33, :2])

            # print(vis_img.shape)
            cv2.imshow(window_name, vis_img)
        #
        if cv2.waitKey(10) == ord('q'): break


def funcDemo(image):
    '''overall process to implement the blaze pose'''
    # 1.update the cfg
    cfg['root_dir'] = root_path
    # 2.init the model
    pose_model = BlazePose(cfg)
    detection_model = PoseDetection(cfg)

    # 3.detection
    norm_roi = detection_model(image)  # normalized roi

    # 4.transform rect
    transfrom_roi = img_process_tools.TransformNormalizedRect(norm_roi,image.shape[1],image.shape[0])
        


# img = cv2.imread(os.path.join(root_path, 'data/beauity_motion.jpg'))
#
# # singleImageDemo(img)
# webCamDemo()
# 1623983605589_video_camera_CurtsyLungetoBalanceRight.mp4
