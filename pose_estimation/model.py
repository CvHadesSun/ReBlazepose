# -*- coding:utf-8 -*-
'''To implment the model forward function
@author:cvhadessun
date:2021-7-6-10:25'''
import os
import cv2
import numpy as np
import time
import math
import tensorflow as tf
from tqdm import tqdm
from utils.img_process_tools import landmarkProjection, ImageTransform, computRotation, ROI


def Sigmoid(value):
    '''implement the sigmoid function'''
    return 1.0 / (1.0 + math.exp(-value))


def heatmap2Joints(heatmap):
    # heatmap:[h,w]
    h, w = heatmap.shape
    idx = np.argmax(heatmap)
    y = int(idx / w) + 0.5
    x = (idx % w - 1) + 0.5
    return x / w, y / h


class BlazePose:
    def __init__(self, config):
        # print('test')
        self.cfg = config
        self.num_landmarks = self.cfg['Landmarks'][3]['num_landmark']
        self.model = self._loadModel()

    def _loadModel(self):
        print('>>init...')
        model_dir = os.path.join(self.cfg['root_dir'],
                                 self.cfg['model_dir'],
                                 self.cfg['Landmarks'][1]['model'])
        num_model = 1
        for i in tqdm(range(num_model), desc='model initialize...'):
            model = tf.lite.Interpreter(model_path=model_dir)
            model.allocate_tensors()
        self.input_details = model.get_input_details()
        self.output_details = model.get_output_details()
        return model

    def printInputInfo(self):
        print('>>Input info:')
        print(self.input_details)
        print('>>Input info print done!')

    def printOutputInfo(self):

        print('>>Output info:')
        print(self.output_details)
        print('>>Output info print done!')

    def _preprocess(self, img):
        input_shape = self.cfg['Landmarks'][0]['input'][0]['shape']
        self.input_shape = input_shape
        value_min = self.cfg['Landmarks'][0]['input'][1]['min']
        value_max = self.cfg['Landmarks'][0]['input'][2]['max']
        assert input_shape[2] == 3
        assert input_shape[0] == input_shape[1]

        img = cv2.resize(img, (input_shape[0], input_shape[1]))  # resize
        #
        np_img = np.array(img).astype(np.float32)
        np_img = np.expand_dims(np_img, 0)
        value_range = np.array([float(value_min), float(value_max)]).astype(np.float32)
        assert value_range[0] < value_range[1]
        scale_img = value_range[0] + (value_range[1] - value_range[0]) * np_img / 255

        return scale_img

    def _preprocessByRoI(self, img, roi):
        input_shape = self.cfg['Landmarks'][0]['input'][0]['shape']
        self.input_shape = input_shape
        value_min = self.cfg['Landmarks'][0]['input'][1]['min']
        value_max = self.cfg['Landmarks'][0]['input'][2]['max']
        assert input_shape[2] == 3
        assert input_shape[0] == input_shape[1]

        scale_img = ImageTransform(img, roi, [input_shape[0], input_shape[1]], [value_min, value_max])

        return scale_img

    def __call__(self, *args, **kwargs):
        input = self._preprocessByRoI(kwargs['img'], kwargs['roi'])
        self.model.set_tensor(self.input_details[0]['index'], input)
        self.model.invoke()
        self.coordinates = self.model.get_tensor(self.output_details[0]['index'])  # [index:497,[1,195]]
        self.pose_flag = self.model.get_tensor(self.output_details[1]['index'])  # output_poseflag index:498 [1,1]
        self.output_segmentation = self.model.get_tensor(
            self.output_details[2]['index'])  # output_segmentation index:486 [1,128,128,1]
        self.output_heatmap = self.model.get_tensor(
            self.output_details[3]['index'])  # output_heatmap index:493 [1,64,64,39]
        self.world_3d = self.model.get_tensor(self.output_details[4]['index'])  # world_3d index:499 [1,117]
        # return self._heatmap2Coordinates()
        np_coordinates = np.squeeze(self.coordinates).reshape(-1, 5)
        norm_joints = np.zeros_like(np_coordinates)
        norm_joints[:, 0] = np_coordinates[:, 0] / self.input_shape[0]
        norm_joints[:, 1] = np_coordinates[:, 1] / self.input_shape[1]
        norm_joints[:, 2] = np_coordinates[:, 2] / self.input_shape[0]
        norm_joints[:, 3] = 1.0 / (1 + np.exp(-np_coordinates[:, 3]))
        norm_joints[:, 4] = 1.0 / (1 + np.exp(-np_coordinates[:, 4]))
        self.norm_joints = norm_joints
        normed_landmarks = norm_joints
        if self.cfg['Landmarks'][4]['refine_landmarks']:
            normed_landmarks = self._refineLandmarkByHeatmap()

        projected_landmarks = self.projectLandmarks(normed_landmarks, kwargs['roi'])
        roi = self._getRoi(projected_landmarks, kwargs['img'].shape[1], kwargs['img'].shape[0])
        return projected_landmarks, roi

    def _heatmap2Coordinates(self):
        new_coordinates = []
        for index in range(self.num_landmarks):
            new_x, new_y = heatmap2Joints(self.output_heatmap[0, :, :, index])
            new_coordinates.append([new_x, new_y])
        return np.array(new_coordinates)

    def _refineLandmarkByHeatmap(self):
        '''using the heatmap to refine the output of coordinates,
        implement this function refer to google/mediapipe:
        mediapipe/mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.cc
        '''
        hm_shape = self.output_heatmap.shape
        assert hm_shape[0] == 1
        assert hm_shape[-1] == self.num_landmarks

        hm_h = hm_shape[1]
        hm_w = hm_shape[2]
        hm_c = hm_shape[3]
        kernel_size = int(self.cfg['Landmarks'][5]['kernel_size'])
        min_confidence_to_refine = float(self.cfg['Landmarks'][6]['min_confidence_to_refine'])
        refine_presence = self.cfg['Landmarks'][7]['refine_presence']
        refine_visibility = self.cfg['Landmarks'][8]['refine_visibility']
        # print(min_confidence_to_refine)
        offset = int((kernel_size - 1) / 2)
        raw_hm = self.output_heatmap[0].reshape(-1)

        refine_coordinates = self.norm_joints

        for lm_index in range(self.num_landmarks):
            center_col = int(self.norm_joints[lm_index, 0] * hm_w)
            # print(center_col>=hm_w)
            center_row = int(self.norm_joints[lm_index, 1] * hm_h)
            if center_col < 0 or center_col >= hm_w or center_row < 0 or center_row >= hm_h:
                continue

            #

            _begin_col = max(0, center_col - offset)
            _end_col = min(hm_w, center_col + offset + 1)
            _begin_row = max(0, center_row - offset)
            _end_row = min(hm_h, center_row + offset + 1)
            _sum = 0.0
            _weighted_col = 0.0
            _weighted_row = 0.0
            _max_confidence_value = 0.0

            for row in range(_begin_row, _end_row):
                for col in range(_begin_col, _end_col):
                    idx = hm_w * hm_c * row + hm_c * col + lm_index
                    confidence = Sigmoid(raw_hm[idx])
                    _sum += confidence
                    _max_confidence_value = max(_max_confidence_value, confidence)
                    _weighted_row = _weighted_row + row * confidence
                    _weighted_col = _weighted_col + col * confidence

            if _max_confidence_value >= min_confidence_to_refine and _sum > 0:
                # print("before",refine_coordinates[lm_index,:2])
                refine_coordinates[lm_index, 0] = _weighted_col / hm_w / _sum
                refine_coordinates[lm_index, 1] = _weighted_row / hm_h / _sum
                # print(lm_index)
                # print("after",refine_coordinates[lm_index, :2])
                # refine_coordinates[lm_index,2] = self.norm_joints[lm_index,2]
                # refine_coordinates[lm_index, 3] = self.norm_joints[lm_index,3]
                # refine_coordinates[lm_index, 4] = self.norm_joints[lm_index, 4]

            if refine_presence and _sum > 0 and self.norm_joints[lm_index, 3] > 0:
                presence = min(self.norm_joints[lm_index, 3], _max_confidence_value)
                refine_coordinates[lm_index, 3] = presence

            if refine_visibility and _sum > 0 and self.norm_joints[lm_index, 4] > 0:
                visibility = min(self.norm_joints[lm_index, 4], _max_confidence_value)
                refine_coordinates[lm_index, 4] = visibility

        return refine_coordinates

    def _getRoi(self, normlandmarks, w, h):
        hip_x = normlandmarks[33, 0] * w
        hip_y = normlandmarks[33, 1] * h
        #
        encode_full_x = normlandmarks[34, 0] * w
        encode_full_y = normlandmarks[34, 1] * h

        size_roi = 2 * (math.sqrt((hip_x - encode_full_x) ** 2 + (hip_y - encode_full_y) ** 2))

        rot = computRotation([hip_x, hip_y], [encode_full_x, encode_full_y],
                             self.cfg['Tracking'][0]['target_angle'])

        return ROI(hip_x / w, hip_y / h, size_roi / w, size_roi / h, rot)

    def projectLandmarks(self, norm_landmarks, roi):

        new_norm_landmarks = np.zeros_like(norm_landmarks)
        for i in range(norm_landmarks.shape[0]):
            new_norm_landmarks[i, :] = landmarkProjection(norm_landmarks[i], roi)

        return new_norm_landmarks

# coordinates = interpreter.get_tensor(output_details[0]['index'])  #[index:497,[1,195]]
# #
# pose_flag = interpreter.get_tensor(output_details[1]['index']) # output_poseflag index:498 [1,1]
# output_segmentation = interpreter.get_tensor(output_details[2]['index']) #output_segmentation index:486 [1,128,128,1]
# output_heatmap = interpreter.get_tensor(output_details[3]['index']) #output_heatmap index:493 [1,64,64,39]
# world_3d = interpreter.get_tensor(output_details[4]['index']) #world_3d index:499 [1,117]


# model_path1 = './../models/pose_landmark_lite.tflite'
# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path=model_path1)
# # interpreter = tflite.Interpreter(model_path=model_path)
# # interpreter =  tf.contrib.lite.Interpreter(model_path=model_path1)
# interpreter.allocate_tensors()
#
#
# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# print(str(input_details))
# output_details = interpreter.get_output_details()
# print(str(output_details))
#
# #output detail
# # [{'name': 'ld_3d', 'index': 497, 'shape': array([  1, 195], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)},
# # {'name': 'output_poseflag', 'index': 498, 'shape': array([1, 1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)},
# # {'name': 'output_segmentation', 'index': 486, 'shape': array([  1, 128, 128,   1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)},
# # {'name': 'output_heatmap', 'index': 493, 'shape': array([ 1, 64, 64, 39], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)},
# # {'name': 'world_3d', 'index': 499, 'shape': array([  1, 117], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
#
# _input = np.random.randn(1,256,256,3).astype(np.float32)
# interpreter.set_tensor(input_details[0]['index'], _input)
# #
# interpreter.invoke()
# #
# coordinates = interpreter.get_tensor(output_details[0]['index'])  #[index:497,[1,195]]
# #
# pose_flag = interpreter.get_tensor(output_details[1]['index']) # output_poseflag index:498 [1,1]
# output_segmentation = interpreter.get_tensor(output_details[2]['index']) #output_segmentation index:486 [1,128,128,1]
# output_heatmap = interpreter.get_tensor(output_details[3]['index']) #output_heatmap index:493 [1,64,64,39]
# world_3d = interpreter.get_tensor(output_details[4]['index']) #world_3d index:499 [1,117]
#
#
# #vis
#
# norm_joints = np.squeeze(coordinates).reshape(-1,5)
#
# norm_joints_xy = norm_joints[:,:2]
# # print(norm_joints)
# img = cv2.imread('./beauity_motion.jpg')
# draw2DJoint(img,norm_joints_xy)
# # visJoints(norm_joints[:33,:])
