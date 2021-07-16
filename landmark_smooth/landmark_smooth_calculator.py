'''to implement the landmark filter calculator
refering to:mediapipe/mediapipe/calculators/util/landmarks_smoothing_calculator.cc
@author:cvhadessun
date:2021-7-13:16:03
'''

import numpy as np
from .filter import RelativeVelocityFilter
from .filter import OneEuroFilter
# from .filter2 import OneEuroFilter
'''
=========================================
=====aux_landmarks smoothing config======
=========================================
filter_name:one_euro_filter
min_cutoff = 0.01
beta = 10.0
derivate_cutoff = 1.0
'''
'''referring to:mediapipe/mediapipe/calculators/util/landmarks_smoothing_calculator.cc'''


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


class LandmarsFilter:
    def Reset(self):
        raise NotImplemented

    def Apply(self, in_landmarks, timestamp, oject_scale_opt):
        '''timestamp uint is: s'''
        raise NotImplemented


class VelocityFilter(LandmarsFilter):
    def __init__(self, window_size=5,
                 velocity_scale=10.0,
                 min_allowed_object_scale=1e-6,
                 disable_value_scaling=False):

        self.window_size_ = window_size
        self.velocity_scale_ = velocity_scale
        self.min_allowed_object_scale_ = min_allowed_object_scale
        self.disable_value_scaling_ = disable_value_scaling
        # init the filters
        self.x_filters_ = []
        self.y_filters_ = []
        self.z_filters_ = []

    def Reset(self):
        self.x_filters_ = []
        self.y_filters_ = []
        self.z_filters_ = []

    # def Apply(self, in_landmarks, timestamp, norm_roi, input_width, input_height):
    def Apply(self, in_landmarks, timestamp, oject_scale_opt):
        value_scale = 1.0
        if not self.disable_value_scaling_:
            # object_scale = GetObjectScaleFromNormROI(norm_roi, input_width, input_height)
            object_scale = oject_scale_opt
            if object_scale < self.min_allowed_object_scale_:
                return in_landmarks
            value_scale = 1.0 / object_scale
        # Initialize filters once.
        self._InitializeFiltersIfEmpty(in_landmarks.shape[0])

        # filter the values
        output_landmarks = in_landmarks
        for i in range(in_landmarks.shape[0]):
            output_landmarks[i, 0] = self.x_filters_[i].Apply(timestamp, value_scale, in_landmarks[i, 0])
            output_landmarks[i, 1] = self.y_filters_[i].Apply(timestamp, value_scale, in_landmarks[i, 1])
            output_landmarks[i, 2] = self.z_filters_[i].Apply(timestamp, value_scale, in_landmarks[i, 2])

        return output_landmarks

    def _InitializeFiltersIfEmpty(self, num_landmarks):
        if not len(self.x_filters) == 0:
            return

        self.x_filters = [RelativeVelocityFilter(self.window_size_, self.velocity_scale_) \
                          for _ in range(num_landmarks)]
        self.y_filters = [RelativeVelocityFilter(self.window_size_, self.velocity_scale_) \
                          for _ in range(num_landmarks)]
        self.z_filters = [RelativeVelocityFilter(self.window_size_, self.velocity_scale_) \
                          for _ in range(num_landmarks)]
        return


class OneEuroFilterImpl(LandmarsFilter):
    def __init__(self, frequency,
                 min_cutoff,
                 beta,
                 derivate_cutoff,
                 min_allowed_object_scale,
                 disable_value_scaling):
        self.frequency_ = frequency
        self.min_cutoff_ = min_cutoff
        self.beta_ = beta
        self.derivate_cutoff_ = derivate_cutoff
        self.min_allowed_object_scale_ = min_allowed_object_scale
        self.disable_value_scaling_ = disable_value_scaling

        self.x_filters_ = []
        self.y_filters_ = []
        self.z_filters_ = []

    def Reset(self):
        self.x_filters_ = []
        self.y_filters_ = []
        self.z_filters_ = []

    # def Apply(self, in_landmarks, timestamp, norm_roi, input_width, input_height):
    def Apply(self, in_landmarks, timestamp, oject_scale_opt):
        value_scale = 1.0
        if not self.disable_value_scaling_:
            object_scale = oject_scale_opt
            if object_scale < self.min_allowed_object_scale_:
                return in_landmarks
            value_scale = 1.0 / object_scale
        # Initialize filters once.
        self._InitializeFiltersIfEmpty(in_landmarks.shape[0])

        # filter the values
        output_landmarks = in_landmarks
        for i in range(in_landmarks.shape[0]):
            output_landmarks[i, 0] = self.x_filters_[i].Apply(timestamp, value_scale, in_landmarks[i, 0])
            output_landmarks[i, 1] = self.y_filters_[i].Apply(timestamp, value_scale, in_landmarks[i, 1])
            # output_landmarks[i, 2] = self.z_filters_[i].Apply(timestamp, value_scale, in_landmarks[i, 2])

        return output_landmarks

    def _InitializeFiltersIfEmpty(self, num_landmarks):
        if not len(self.x_filters_) == 0:
            return
        # print("initial...")
        self.x_filters_ = [OneEuroFilter(self.frequency_, self.min_cutoff_, self.beta_, self.derivate_cutoff_) \
                           for _ in range(num_landmarks)]
        self.y_filters_ = [OneEuroFilter(self.frequency_, self.min_cutoff_, self.beta_, self.derivate_cutoff_) \
                           for _ in range(num_landmarks)]
        self.z_filters_ = [OneEuroFilter(self.frequency_, self.min_cutoff_, self.beta_, self.derivate_cutoff_) \
                           for _ in range(num_landmarks)]
        return

# input landmark is input scale, and timestamp uint is s.
