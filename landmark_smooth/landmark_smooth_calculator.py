'''to implement the landmark filter calculator
refering to:mediapipe/mediapipe/calculators/util/landmarks_smoothing_calculator.cc
@author:cvhadessun
date:2021-7-13:16:03
'''

import numpy as np

'''
=========================================
=====aux_landmarks smoothing config======
=========================================
filter_name:one_euro_filter
min_cutoff = 0.01
beta = 10.0
derivate_cutoff = 1.0
'''


def NormalizedLandmarksToLandmarks(norm_landmarks, input_width, input_height):
    '''transform the normalized landmarks to the input image scale'''
    landmarks = norm_landmarks
    landmarks[:, 0] = norm_landmarks[:, 0] * input_width
    landmarks[:, 1] = norm_landmarks[:, 1] * input_height
    landmarks[:, 2] = norm_landmarks[:, 2] * input_width
    return landmarks


def LandmarksToNormalizedLandmarks(landmarks, input_widht, input_height):
    '''transform the landmarks into normalized scale with input image shape'''
    norm_landmarks = landmarks
    norm_landmarks[:, 0] = landmarks[:, 0] / input_widht
    norm_landmarks[:, 1] = landmarks[:, 1] / input_height
    norm_landmarks[:, 2] = landmarks[:, 2] / input_widht

    return norm_landmarks


def GetObjectScaleFromLandmark(landmarks):
    '''get the object scale using landmarks from original input'''
    xmin = np.min(landmarks[:, 0])
    xmax = np.max(landmarks[:, 0])

    ymin = np.min(landmarks[:, 1])
    ymax = np.max(landmarks[:, 1])

    object_width = xmax - xmin
    object_height = ymax - ymin

    return (object_height + object_width) / 2.0


def GetObjectScaleFromNormROI(norm_roi, input_width, input_height):
    '''get the object scale using normalized roi and input size'''
    object_width = norm_roi.width * input_width
    object_height = norm_roi.height * input_height

    return (object_width + object_height) / 2.0


def GetObjectScaleFromROI(roi):
    '''get the object scale with no normalized roi. '''
    return (roi.width + roi.height) / 2.0


class VelocityFilter:
    def __init__(self, window_size=5,
                 velocity_scale=10.0,
                 min_allowed_object_scale=1e-6,
                 disable_value_scaling=False):
        self.window_size = window_size
        self.velocity_scale = velocity_scale
        self.min_allowed_object_scale = min_allowed_object_scale
        self.disable_value_scaling = disable_value_scaling

    def Reset(self):
        pass

    def __call__(self, *args, **kwargs):
        value_scale = 1.0
        norm_roi = kwargs['norm_roi']
        input_width = kwargs['w']
        input_height = kwargs['h']
        input = kwargs['input']
        if not self.disable_value_scaling:
            object_scale = GetObjectScaleFromNormROI(norm_roi, input_width, input_height)
            if object_scale < self.min_allowed_object_scale:
                return input
            value_scale = 1.0/ object_scale

