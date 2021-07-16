'''A demo to test the pose estimation using BlazePose landmarks detection model
@author:cvhades
date:2021-7-6-11:25'''
import os
import cv2
import sys
import numpy as np
import time

from config.cfg import cfg
from pose_estimation.model import BlazePose
from pose_detection.model import PoseDetection
from utils import tools
from utils import img_process_tools

from landmark_smooth import filter_tools
from landmark_smooth.landmark_smooth_calculator import GetObjectScaleFromNormROI

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
    pose_threshold = 0.5
    coordinates = np.zeros([39, 5])
    # init the filters
    aux_vis_filter, aux_filter, vis_landmark_filter, landmark_filter = initFilter(cfg)
    i = -1
    t0 = time.time()
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
                    tracking = True  # open the tracking branch
                    print("detection++", pose_model.pose_flag[0][0])
                else:  # pose result is low confidence
                    coordinates[:, :] = 0
                    transform_roi = img_process_tools.ROI(0.5, 0.5, 1., 1., 0.)  # the whole frame
                    tracking = False  # close the tracking branch and go on detection

            # post the landmarks to original scale

            new_coordinates = pose_model.projectLandmarks(coordinates, transform_roi)
            # print(new_coordinates.shape)
            # landmarks filtering
            # get the object scale
            # object_scale = GetObjectScaleFromNormROI(transform_roi,frame.shape[1],frame.shape[0])
            timestamp = time.time()
            # print(timestamp)
            # aux_landmarks filtering
            tmp_roi = pose_model.getRoi([new_coordinates[33, 0], new_coordinates[33, 1]],
                                             [new_coordinates[34, 0], new_coordinates[34, 1]],
                                             frame.shape[1], frame.shape[0])

            filtered_aux_landmarks, aux_vis_filter, aux_filter = filter_tools.FilterAuxLandmarks(aux_vis_filter,
                                                                                                 aux_filter,
                                                                                                 new_coordinates[33:35,
                                                                                                 :],
                                                                                                 timestamp,
                                                                                                 frame.shape[1],
                                                                                                 frame.shape[0],
                                                                                                 tmp_roi)

            filtered_landmarks, vis_landmark_filter, landmark_filter = filter_tools.FitlerLandmarks(vis_landmark_filter,
                                                                                                    landmark_filter,
                                                                                                    new_coordinates[:33,
                                                                                                    :],
                                                                                                    timestamp,
                                                                                                    frame.shape[1],
                                                                                                    frame.shape[0],
                                                                                                    tmp_roi)

            # new_coordinates[:33,:] = filtered_landmarks[:,:]

            # using landmarks results to locate the roi

            # bbox = img_process_tools.landmarkToDetection(new_coordinates)

            pre_pose_roi = pose_model.getRoi([filtered_aux_landmarks[0, 0], filtered_aux_landmarks[0, 1]],
                                             [filtered_aux_landmarks[1, 0], filtered_aux_landmarks[1, 1]],
                                             frame.shape[1], frame.shape[0])

            # pre_pose_roi.set_x(bbox[0]+bbox[2]/2)
            # pre_pose_roi.set_y(bbox[1]+bbox[3]/2)
            # pre_pose_roi.set_width(bbox[2])
            # pre_pose_roi.set_height(bbox[3])

            cv2.circle(frame,
                       (int(filtered_aux_landmarks[0, 0] * frame.shape[1]), int(filtered_aux_landmarks[0, 1] * frame.shape[0])), 5,
                       (0, 255, 255), -1)
            cv2.circle(frame,
                       (int(filtered_aux_landmarks[1, 0] * frame.shape[1]), int(filtered_aux_landmarks[1, 1] * frame.shape[0])), 5,
                       (0, 255, 255), -1)
            # visualization:
            vis_img = tools.draw2DJoint(frame, filtered_landmarks[:, :2])
            cv2.imshow(window_name, vis_img)
        # cv2.waitKey()
        if cv2.waitKey(1)==ord('q'):break


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
    print(transfrom_roi.rotation)

    # 6. pose estimation
    pose_model(img=image, roi=transfrom_roi)
    coord = pose_model.postLandmarks()
    projected_coord = pose_model.projectLandmarks(coord, transfrom_roi)

    # 7. filtering
    # split the coordinates into landmarks and aux_landmarks
    in_landmarks = projected_coord[:33, :]
    aux_landmars = projected_coord[33:35, :]

    # a. filter the aux_landmarks

    vis_img = tools.draw2DJoint(image, projected_coord[:33, :2])

    cv2.imshow('show', vis_img)
    cv2.waitKey()
    # cv2.imwrite('./data/beauity_motion_pose.jpg',vis_img)


def initFilter(cfg):
    aux_vis_filter = filter_tools.initVisFilter(cfg['Filter'][0]['Visbility_filter'][0]['filter_name'],
                                                (cfg['Filter'][0]['Visbility_filter'][1]['parameters'][0]['alpha']))

    aux_landmark_filter = filter_tools.initOneEuroFilter(cfg['Filter'][1]['Aux_landmark_filter'][0]['filter_name'],
                                                         (cfg['Filter'][1]['Aux_landmark_filter'][1]['parameters'][0][
                                                              'frequency'],
                                                          cfg['Filter'][1]['Aux_landmark_filter'][1]['parameters'][1][
                                                              'min_cutoff'],
                                                          cfg['Filter'][1]['Aux_landmark_filter'][1]['parameters'][2][
                                                              'beta'],
                                                          cfg['Filter'][1]['Aux_landmark_filter'][1]['parameters'][3][
                                                              'derivate_cutoff'],
                                                          cfg['Filter'][1]['Aux_landmark_filter'][1]['parameters'][4][
                                                              'min_allowed_object_scale'],
                                                          cfg['Filter'][1]['Aux_landmark_filter'][1]['parameters'][5][
                                                              'disable_value_scaling'])
                                                         )

    landmark_vis_filter = filter_tools.initVisFilter(cfg['Filter'][2]['Landmark_filter'][0]['filter_name'],
                                                     (cfg['Filter'][2]['Landmark_filter'][1]['parameters'][0]['alpha']))

    landmark_filter = filter_tools.initOneEuroFilter(cfg['Filter'][2]['Landmark_filter'][2]['filter_name'],
                                                     (cfg['Filter'][2]['Landmark_filter'][3]['parameters'][0][
                                                          'frequency'],
                                                      cfg['Filter'][2]['Landmark_filter'][3]['parameters'][1][
                                                          'min_cutoff'],
                                                      cfg['Filter'][2]['Landmark_filter'][3]['parameters'][2]['beta'],
                                                      cfg['Filter'][2]['Landmark_filter'][3]['parameters'][3][
                                                          'derivate_cutoff'],
                                                      cfg['Filter'][2]['Landmark_filter'][3]['parameters'][4][
                                                          'min_allowed_object_scale'],
                                                      cfg['Filter'][2]['Landmark_filter'][3]['parameters'][5][
                                                          'disable_value_scaling'])
                                                     )

    return aux_vis_filter, aux_landmark_filter, landmark_vis_filter, landmark_filter


# img = cv2.imread(os.path.join(root_path, 'data/test.png'))
#
# funcDemo(img)

# img = cv2.imread(os.path.join(root_path, 'data/beauity_motion.jpg'))
#
# # singleImageDemo(img)
webCamDemo()
# 1623983605589_video_camera_CurtsyLungetoBalanceRight.mp4
