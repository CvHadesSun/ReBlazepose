import os
import cv2
import tensorflow as tf
import numpy as np
import torch
model_path1 = './../pose_detection.tflite'
dir_img = './../beauity_motion.jpg'

def _preprocess(x):
    """Converts the image pixels to the range [-1, 1]."""

    return x / 127.5 - 1.0

def preprocess(img_dir):
    img = cv2.imread(img_dir)
    # resize:
    img = cv2.resize(img, (128, 128))
    np_img = np.array(img).astype(np.float32)
    np_img = np.expand_dims(np_img, 0)
    return np_img / 127.5 - 1.0




def ProcessInput(input,tf_model_dir):
    '''#input
    [{'name': 'input', 'index': 0, 'shape': array([  1, 128, 128,   3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
    #output
    [{'name': 'regressors', 'index': 266, 'shape': array([  1, 896,  12], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)},
    {'name': 'classificators', 'index': 265, 'shape': array([  1, 896,   1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}'''

    interpreter = tf.lite.Interpreter(model_path=tf_model_dir)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #input
    _input = preprocess(input)
    interpreter.set_tensor(input_details[0]['index'], _input)
    interpreter.invoke()

    #output
    output_reg = interpreter.get_tensor(output_details[0]['index'])
    output_cls = interpreter.get_tensor(output_details[1]['index'])


    return output_reg,output_cls

def load_anchors(path,device_='cpu'):
    anchors = torch.tensor(np.load(path), dtype=torch.float32, device=device_)
    return  anchors

def decode_boxes(raw_boxes, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    x_scale = 128.0
    y_scale = 128.0
    h_scale = 128.0
    w_scale = 128.0
    # min_score_thresh = 0.75


    #min_suppression_threshold = 0.3

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

def rawOutput2Detection(output_cls,output_reg,anchors):
    '''convert the output of network to detection results'''
    #
    min_score_thresh=0.5
    raw_box_tensor = torch.from_numpy(output_reg)
    raw_score_tensor = torch.from_numpy(output_cls)
    detection_boxes = decode_boxes(raw_box_tensor, anchors)
    score_clipping_thresh = 100.0
    raw_score_tensor = raw_score_tensor.clamp(-score_clipping_thresh, score_clipping_thresh)
    detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
    mask = detection_scores >= min_score_thresh
    #
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
        output_detections.append(torch.cat((boxes, scores), dim=-1))

    return output_detections

def weighted_non_max_suppression(detections):
    '''implement Non-maximum suppression to remove overlapping detections'''


    if len(detections) == 0: return []
    min_suppression_threshold = 0.3 #threshold of  NMS

    output_detections = []

    # Sort the detections from highest to lowest score.
    remaining = torch.argsort(detections[:, 12], descending=True)

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.clone()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :12]
            scores = detections[overlapping, 12:13]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(dim=0) / total_score
            weighted_detection[:12] = weighted
            weighted_detection[12] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


def main(img_path,tf_model_path,anchor_path):
    '''to detect the human in the img'''
    reg,cls = ProcessInput(img_path,tf_model_path)
    anchors = load_anchors(anchor_path)

    #process output from network
    detections = rawOutput2Detection(cls,reg,anchors)

    # 4. Non-maximum suppression to remove overlapping detections:
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 13))
        filtered_detections.append(faces)

    return filtered_detections


# if __name__ == '__main__':
#     detections=main('./../beauity_motion.jpg','./../pose_detection.tflite','./../anchors.npy')
#
#     print(detections)




