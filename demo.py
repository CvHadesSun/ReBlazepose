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
    tracking = False
    #
    pre_pose_roi = img_process_tools.ROI(0., 0., 0., 0., 0.)
    detection_roi = img_process_tools.ROI(0.5, 0.5, 1.0, 1.0, 0.)  # the whole frame
    transform_roi = img_process_tools.ROI(0.5, 0.5, 1., 1., 0.)  # the whole frame
    pose_threshold = 0.6
    coordinates = np.zeros([39, 5])
    while True:
        ret, frame = cap.read()
        if ret:
            if tracking:
                transform_roi = img_process_tools.TransformNormalizedRect(pre_pose_roi, frame.shape[1], frame.shape[0])
                pose_model(img=frame, roi=transform_roi)  # pose detection
                if pose_model.pose_flag[0][0] > pose_threshold:  # poseflagis in [0,1]
                    coordinates = pose_model.postLandmarks()  # normalized coordinates
                    tracking = True  # open the tracking branch
                    print("tracking+", pose_model.pose_flag[0][0])
                else:  # pose result is low confidence
                    coordinates[:, :] = 0
                    transform_roi = img_process_tools.ROI(0.5, 0.5, 1., 1., 0.)  # the whole frame
                    tracking = False  # close the tracking branch and go on detection


            else:

                detection_roi = detection_model(frame)
                transform_roi = img_process_tools.TransformNormalizedRect(detection_roi, frame.shape[1], frame.shape[0])
                pose_model(img=frame, roi=transform_roi)  # pose detection
                if pose_model.pose_flag[0][0] > pose_threshold:  # poseflagis in [0,1]
                    coordinates = pose_model.postLandmarks()  # normalized coordinates
                    tracking = False  # open the tracking branch
                    print("detection++", pose_model.pose_flag[0][0])
                else:  # pose result is low confidence
                    coordinates[:, :] = 0
                    transform_roi = img_process_tools.ROI(0.5, 0.5, 1., 1., 0.)  # the whole frame
                    tracking = False  # close the tracking branch and go on detection

            # post the landmarks to original scale

            new_coordinates = pose_model.projectLandmarks(coordinates, transform_roi)

            # using landmarks results to locate the roi

            # bbox = img_process_tools.landmarkToDetection(new_coordinates)

            pre_pose_roi = pose_model.getRoi([new_coordinates[33, 0], new_coordinates[33, 1]],
                                             [new_coordinates[34, 0], new_coordinates[34, 1]],
                                             frame.shape[1], frame.shape[0])

            # pre_pose_roi.set_x(bbox[0]+bbox[2]/2)
            # pre_pose_roi.set_y(bbox[1]+bbox[3]/2)
            # pre_pose_roi.set_width(bbox[2])
            # pre_pose_roi.set_height(bbox[3])

            cv2.circle(frame,
                       (int(new_coordinates[33, 0] * frame.shape[1]), int(new_coordinates[33, 1] * frame.shape[0])), 5,
                       (0, 255, 255), -1)
            cv2.circle(frame,
                       (int(new_coordinates[36, 0] * frame.shape[1]), int(new_coordinates[36, 1] * frame.shape[0])), 5,
                       (0, 255, 255), -1)
            # visualization:
            vis_img = tools.draw2DJoint(frame, new_coordinates[:33, :2])
            cv2.imshow(window_name, vis_img)
        cv2.waitKey()
        # if cv2.waitKey(1)==ord('q'):break


def funcDemo(image):
    '''overall process to implement the blaze pose'''
    # 1.update the cfg
    cfg['root_dir'] = root_path
    # 2.init the model
    pose_model = BlazePose(cfg)
    detection_model = PoseDetection(cfg)
    # detection_model.printOutputInfo()

    # 3.detection
    norm_roi = detection_model(image)  # normalized roi

    # 4.transform rect
    transfrom_roi = img_process_tools.TransformNormalizedRect(norm_roi, image.shape[1], image.shape[0])

    # padding is skipped

    # 6. pose estimation
    pose_model(img=image, roi=transfrom_roi)
    coord = pose_model.postLandmarks()
    projected_coord = pose_model.projectLandmarks(coord, transfrom_roi)

    vis_img = tools.draw2DJoint(image, projected_coord[:33, :2])

    cv2.imshow('show', vis_img)
    cv2.waitKey()
    # cv2.imwrite('./data/beauity_motion_pose.jpg',vis_img)


#
img = cv2.imread(os.path.join(root_path, 'data/beauity_motion.jpg'))

funcDemo(img)

# img = cv2.imread(os.path.join(root_path, 'data/beauity_motion.jpg'))
#
# # singleImageDemo(img)
# webCamDemo()
# 1623983605589_video_camera_CurtsyLungetoBalanceRight.mp4
