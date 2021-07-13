import cv2
import tensorflow as tf
import numpy as np
import torch
import os
from tqdm import tqdm
import math
from config.cfg import cfg
from utils.nms import *
from utils.img_process_tools import ROI, getRoi, getRotation, computRotation


class PoseDetection:
    def __init__(self, config):
        self.cfg = config
        self.input_width = self.cfg['PoseDetection'][0]['input'][0]['shape'][1]
        self.input_height = self.cfg['PoseDetection'][0]['input'][0]['shape'][0]
        # self.ssd_option = self._loadSsdOption()
        # 1.init the model
        self.model = self._loadModel()
        # 2.load the ssd anchors
        self._loadSSDAnchors()

    def _preprocess(self, img):
        value_min = self.cfg['PoseDetection'][0]['input'][1]['min']
        value_max = self.cfg['PoseDetection'][0]['input'][2]['max']

        img = cv2.resize(img, (self.input_width, self.input_height))  # resize
        #
        np_img = np.array(img).astype(np.float32)
        np_img = np.expand_dims(np_img, 0)
        value_range = np.array([float(value_min), float(value_max)]).astype(np.float32)
        assert value_range[0] < value_range[1]
        scale_img = value_range[0] + (value_range[1] - value_range[0]) * np_img / 255

        return scale_img

    def _loadSsdOption(self):
        '''to laod the ssd anchors options '''
        options = {}
        options["num_layers"] = self.cfg["PoseDetection"][2]["ssd_option"][0]["num_layers"]
        options["min_scale"] = self.cfg["PoseDetection"][2]["ssd_option"][1]["min_scale"]
        options["max_scale"] = self.cfg["PoseDetection"][2]["ssd_option"][2]["max_scale"]
        options["input_size_height"] = self.cfg["PoseDetection"][2]["ssd_option"][3]["input_size_width"]
        options["input_size_width"] = self.cfg["PoseDetection"][2]["ssd_option"][4]["input_size_height"]
        options["anchor_offset_x"] = self.cfg["PoseDetection"][2]["ssd_option"][5]["anchor_offset_x"]
        options["anchor_offset_y"] = self.cfg["PoseDetection"][2]["ssd_option"][6]["anchor_offset_y"]
        options["strides"] = self.cfg["PoseDetection"][2]["ssd_option"][7]["strides"]
        options["aspect_ratios"] = self.cfg["PoseDetection"][2]["ssd_option"][8]["aspect_ratios"]
        options["reduce_boxes_in_lowest_layer"] = self.cfg["PoseDetection"][2]["ssd_option"][9][
            "reduce_boxes_in_lowest_layer"]
        options["interpolated_scale_aspect_ratio"] = self.cfg["PoseDetection"][2]["ssd_option"][10][
            "interpolated_scale_aspect_ratio"]
        options["fixed_anchor_size"] = self.cfg["PoseDetection"][2]["ssd_option"][11]["fixed_anchor_size"]

        return options

    def _loadDecodeOptions(self):
        pass

    def _loadModel(self):
        print('>>init...')
        model_dir = os.path.join(self.cfg['root_dir'],
                                 self.cfg['model_dir'],
                                 self.cfg['PoseDetection'][3]['model'])
        num_model = 1
        for i in tqdm(range(num_model), desc='detection model initialize...'):
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

    def _loadSSDAnchors(self, device_='cpu'):
        anchors_dir = os.path.join(self.cfg['root_dir'],
                                   self.cfg['model_dir'],
                                   self.cfg['PoseDetection'][4]['ssd_anchor'])
        self.anchors = torch.tensor(np.load(anchors_dir), dtype=torch.float32, device=device_)

    def __call__(self, *args, **kwargs):
        '''input the image to detection the human and reture the roi of human location in the image'''
        # input tensor
        h, w, _ = args[0].shape
        input_img = self._preprocess(args[0])
        self.model.set_tensor(self.input_details[0]['index'], input_img)
        self.model.invoke()  # forward

        output_reg = self.model.get_tensor(self.output_details[0]['index'])
        output_cls = self.model.get_tensor(self.output_details[1]['index'])

        #save outputs into the file
        # np.save('./results/detection_model_output_reg.npy',output_reg)
        # np.save('./results/detection_model_output_cls.npy', output_cls)

        detections = self._rawOutput2Detection(output_cls, output_reg)
        # np.save('./results/_rawOutput2Detection_output.npy',detections[0].numpy())

        # Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = weighted_non_max_suppression(detections[i])
            # np.save('./results/nms_output.npy',faces[0].numpy())
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 13))
            filtered_detections.append(faces.numpy())
        self.final_detection = filtered_detections


        # get Roi
        if self.final_detection[0].shape[0] == 0:
            return ROI(0.5, 0.5, 1., 1., 0)

        roi = self._getROI(w, h)[0]  # in fact there is one roi

        return roi

        # return filtered_detections

    def _decode_boxes(self, raw_boxes, anchors):
        x_scale = self.cfg['PoseDetection'][1]['decode'][10]['x_scale']
        y_scale = self.cfg['PoseDetection'][1]['decode'][11]['y_scale']
        w_scale = self.cfg['PoseDetection'][1]['decode'][12]['w_scale']
        h_scale = self.cfg['PoseDetection'][1]['decode'][13]['h_scale']

        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(4):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _rawOutput2Detection(self, output_cls, output_reg):
        min_score_thresh = self.cfg['PoseDetection'][1]['decode'][14]['min_score_thresh']
        score_clipping_thresh = self.cfg['PoseDetection'][1]['decode'][8]['score_clipping_thresh']
        raw_box_tensor = torch.from_numpy(output_reg)
        raw_score_tensor = torch.from_numpy(output_cls)
        detection_boxes = self._decode_boxes(raw_box_tensor, self.anchors)
        # np.save('./results/decode_box_output.npy',detection_boxes.numpy())
        raw_score_tensor = raw_score_tensor.clamp(-score_clipping_thresh, score_clipping_thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
        mask = detection_scores >= min_score_thresh
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _getROI(self, w, h):
        #
        #     return ROI(0,0,0,0,0)
        rois = []
        for d in self.final_detection:
            face_bbox = d[0][:4]
            #
            hip_x = d[0][4] * w
            hip_y = d[0][5] * h
            #
            encode_full_x = d[0][6] * w
            encode_full_y = d[0][7] * h

            # rot = getRotation([hip_x, hip_y], [encode_full_x, encode_full_y], self.cfg['Tracking'][0]['target_angle'])
            #

            rot = computRotation([hip_x, hip_y], [encode_full_x, encode_full_y],
                                 self.cfg['Tracking'][0]['target_angle'])
            mid_should_x = d[0][8]
            mid_should_y = d[0][9]
            #
            encode_up_x = d[0][10]
            encode_up_y = d[0][11]

            size_roi = 2 * (math.sqrt((hip_x - encode_full_x) ** 2 + (hip_y - encode_full_y) ** 2))
            tmp_roi = ROI(hip_x / w, hip_y / h, size_roi / w, size_roi / h, rot)
            rois.append(tmp_roi)
        return rois

# root_path = os.path.abspath(os.path.join('..'))
# cfg['root_dir'] = root_path
#
# pd = PoseDetection(cfg)
#
# pd.printOutputInfo()
# print(len(pd.output_details))
