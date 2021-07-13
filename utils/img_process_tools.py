import math
import numpy as np
import cv2


class ROI:
    def __init__(self, *args):
        self.center_x = 0
        self.center_y = 0
        self.width = 0
        self.height = 0
        self.rotation = 0
        self.set_value(*args)

    def set_value(self, *args):
        x, y, w, h, rot = args[:]
        self.set_x(x)
        self.set_y(y)
        self.set_width(w)
        self.set_height(h)
        self.set_rotation(rot)

    def set_x(self, cx):
        self.center_x = cx

    def set_y(self, cy):
        self.center_y = cy

    def set_width(self, w):
        self.width = w

    def set_height(self, h):
        self.height = h

    def set_rotation(self, rot):
        self.rotation = rot


def getRoi(rect, input_width, input_height, is_norm):
    '''to get the absolute roi in the image,
    refer to: mediapipe/calculators/tensor/image_to_tensor_utils.cc,
              mediapipe/calculators/tensor/image_to_tensor_utils.h'''

    new_rect = ROI(0, 0, 0, 0, 0)
    if is_norm:
        new_rect.set_x(rect.center_x * input_width)
        new_rect.set_y(rect.center_y * input_height)
        new_rect.set_width(rect.width * input_width)
        new_rect.set_height(rect.height * input_height)
        new_rect.set_rotation(rect.rotation)
        return new_rect

    else:
        new_rect.set_x(0.5 * input_width)
        new_rect.set_y(0.5 * input_height)
        new_rect.set_width(input_width)
        new_rect.set_height(input_height)

        return new_rect


def PadRoi(input_width, input_height, keep_aspect_ratio, roi):
    '''using roi to pad the input and get the padding box(letterbox),roi is normalized,
    @return normalized roi and normalized padding bbox,it is letterbox'''
    if keep_aspect_ratio:
        return roi, [0.0, 0.0, 0.0, 0.0]
    assert input_height > 0
    assert input_width > 0

    aspect_ratio = input_height / input_width

    assert roi.width > 0
    assert roi.height > 0
    # the normalized roi to no normalized roi
    roi = getRoi(roi, input_width, input_height, True)

    roi_aspect_ratio = roi.height / roi.width
    print(roi_aspect_ratio)

    vertical_padding = 0.0
    horizontal_padding = 0.0
    new_width = 0
    new_height = 0

    if aspect_ratio > roi_aspect_ratio:
        new_width = roi.width
        new_height = roi.width * aspect_ratio
        vertical_padding = (1.0 - roi_aspect_ratio / aspect_ratio) / 2.0
    else:
        new_width = roi.height / aspect_ratio
        new_height = roi.height
        horizontal_padding = (1.0 - aspect_ratio / roi_aspect_ratio) / 2.0

    roi.width = new_width
    roi.height = new_height

    padding = [horizontal_padding, vertical_padding, horizontal_padding, vertical_padding]  # is normlized padding
    # normalized
    roi.set_x(roi.center_x / input_width)
    roi.set_y(roi.center_y / input_height)
    roi.set_width(roi.width / input_width)
    roi.set_height(roi.height / input_height)
    return roi, padding


def rotate_points(points, center, rot_rad):
    """
    :param points:  N*2
    :param center:  2
    :param rot_rad: scalar
    :return: N*2
    """
    rot_rad = rot_rad * np.pi / 180.0
    rotate_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                           [np.sin(rot_rad), np.cos(rot_rad)]])
    center = center.reshape(2, 1)
    points = points.T
    points = rotate_mat.dot(points - center) + center

    return points.T


def GetRotatedSubRectToRectTransformMatrix(roi, image_width, image_height, flip=False):
    '''to get transform matrix'''

    #  Matrix to convert X,Y to [-0.5, 0.5] range "initial_translate_matrix"
    # { 1.0f,  0.0f, 0.0f, -0.5f}
    # { 0.0f,  1.0f, 0.0f, -0.5f}
    # { 0.0f,  0.0f, 1.0f,  0.0f}
    # { 0.0f,  0.0f, 0.0f,  1.0f}
    a = roi.width
    b = roi.height
    # Matrix to scale X,Y,Z to sub rect "scale_matrix"
    # Z has the same scale as X.
    # {   a, 0.0f, 0.0f, 0.0f}
    # {0.0f,    b, 0.0f, 0.0f}
    # {0.0f, 0.0f,    a, 0.0f}
    # {0.0f, 0.0f, 0.0f, 1.0f}

    fl = -1 if flip else 1
    # Matrix for optional horizontal flip around middle of output image.
    # { fl  , 0.0f, 0.0f, 0.0f}
    # { 0.0f, 1.0f, 0.0f, 0.0f}
    # { 0.0f, 0.0f, 1.0f, 0.0f}
    # { 0.0f, 0.0f, 0.0f, 1.0f}

    c = math.cos(roi.rotation / 180 * math.pi)
    d = math.sin(roi.rotation / 180 * math.pi)
    # Matrix to do rotation around Z axis "rotate_matrix"
    # {    c,   -d, 0.0f, 0.0f}
    # {    d,    c, 0.0f, 0.0f}
    # { 0.0f, 0.0f, 1.0f, 0.0f}
    # { 0.0f, 0.0f, 0.0f, 1.0f}

    e = roi.center_x
    f = roi.center_y
    # Matrix to do X,Y translation of sub rect within parent rect"translate_matrix"
    # {1.0f, 0.0f, 0.0f, e   }
    # {0.0f, 1.0f, 0.0f, f   }
    # {0.0f, 0.0f, 1.0f, 0.0f}
    # {0.0f, 0.0f, 0.0f, 1.0f}

    g = 1. / image_width
    h = 1. / image_height
    # Matrix to scale X,Y,Z to [0.0, 1.0] range "post_scale_matrix"
    # {g,    0.0f, 0.0f, 0.0f}
    # {0.0f, h,    0.0f, 0.0f}
    # {0.0f, 0.0f,    g, 0.0f}
    # {0.0f, 0.0f, 0.0f, 1.0f}

    # matrix

    # row 1
    mat = np.ones([4, 4], dtype=np.float)
    mat[0, 0] = a * c * fl * g
    mat[0, 1] = -b * d * g
    mat[0, 2] = 0.
    mat[0, 3] = (-0.5 * a * c * fl + 0.5 * b * d + e) * g
    # row 2
    mat[1, 0] = a * d * fl * h
    mat[1, 1] = b * c * h
    mat[1, 2] = 0.
    mat[1.3] = (-0.5 * b * c - 0.5 * a * d * fl + f) * h
    # row 3
    mat[2, 0] = 0.
    mat[2, 1] = 0.
    mat[2, 2] = a * g
    mat[2, 3] = 0.
    # row 4
    mat[3, 0] = 0.
    mat[3, 1] = 0.
    mat[3, 2] = 0.
    mat[3, 3] = 1.

    return mat


def ImageTransform(image, roi, output_size, range_):
    '''to transform the roi box image to output size and transform the value from [0,255] to [range_]'''
    # output_width,output_height =output_size
    # range_min_ ,range_max_ = range_
    # image format:RGB (only supported)

    assert len(image.shape) > 0
    assert image.shape[-1] == 3

    roi = getRoi(roi, image.shape[1], image.shape[0], True)
    # roi = TransformRect(roi)

    roi_xmin = roi.center_x - roi.width / 2
    roi_ymin = roi.center_y - roi.height / 2
    roi_xmax = roi.center_x + roi.width / 2
    roi_ymax = roi.center_y + roi.height / 2
    rect_points = np.array([[roi_xmin, roi_ymax],
                            [roi_xmin, roi_ymin],
                            [roi_xmax, roi_ymin],
                            [roi_xmax, roi_ymax]])
    rect_center = np.array([roi.center_x, roi.center_y])
    rot_rect_points = rotate_points(rect_points, rect_center, roi.rotation).astype(np.float32)  # Counterclockwise
    output_height, output_width = output_size

    # print(rot_rect_points.shape)
    dst_corners = np.array([[0.0, output_height],
                            [0.0, 0.0],
                            [output_width, 0.0],
                            [output_width, output_height]]).astype(np.float32)
    projection_matrix = cv2.getPerspectiveTransform(rot_rect_points, dst_corners)

    warped = cv2.warpPerspective(image, projection_matrix, (output_width, output_height),
                                 flags=cv2.INTER_LINEAR)  # padding zeros
    # cv2.imwrite('./results/warped_roi.jpg',warped)
    # scale image value into dst range:[0,255]-->[range_[0],rang_[1]]

    assert len(range_) > 0
    assert range_[0] < range_[1]
    np_img = np.array(warped).astype(np.float32)
    np_img = np.expand_dims(np_img, 0)
    np_warped = np.array(np_img)
    scale_img = range_[0] + (range_[1] - range_[0]) * np_warped / 255.0

    return scale_img


def getRotation(point1, point2, dst_angle):
    '''two points belong a straight line and ,the angle between the straight line and the
    vertical direction：a, and return (90-a)
    @point:[x,y]
    @point1 is the center'''

    vertical_dist = point2[1] - point1[1]
    horizontial_dist = point2[0] - point1[0]
    tan_thelta = math.atan(-vertical_dist / horizontial_dist) / math.pi * 180.0
    if point1[0] <= point2[0]:
        return dst_angle - tan_thelta

    else:
        return -dst_angle - tan_thelta


def computRotation(center, point, target_angle_):
    '''referring to:
     mediapipe/calculators/util/detections_to_rects_calculators.cc
    mediapipe/calculators/util/detections_to_rects_calculators.h
    '''
    target_angle_ = target_angle_ * math.pi / 180.0
    vertical_dist = point[1] - center[1]
    horizontial_dist = point[0] - center[0]
    norm_angle = NormalizeRadians(target_angle_ - math.atan2(-vertical_dist, horizontial_dist))

    return norm_angle / math.pi * 180.0

    # return (target_angle_ - math.atan2(-vertical_dist, horizontial_dist)) * 180.0 / math.pi


def NormalizeRadians(angle_):
    '''referring to: transform the angle into [-pi,pi]
    mediapipe/calculators/util/detections_to_rects_calculators.cc
    mediapipe/calculators/util/detections_to_rects_calculators.h
    '''
    return angle_ - 2 * math.pi * math.floor((angle_ - (-math.pi)) / (2 * math.pi))


def TransformRect(roi):
    '''referring to: the roi is not normalized roi
    mediapipe/calculators/util/rect_transformation_calculator.cc
    '''
    square_long = True
    w = roi.width
    h = roi.height
    rot = roi.rotation
    rot = rot / math.pi * 180.

    x_shift = 0.
    y_shift = 0.
    x_scale = 1.25
    y_scale = 1.25

    if rot == 0.:
        roi.set_x(roi.center_x + x_shift)
        roi.set_y(roi.center_y + y_shift)
    else:
        x_shift = w * x_shift * math.cos(rot) - h * y_shift * math.sin(rot)
        y_shift = w * x_shift * math.sin(rot) + h * y_shift * math.cos(rot)
        roi.set_x(roi.center_x + x_shift)
        roi.set_y(roi.center_y + y_shift)
    if square_long:
        long_side = max(w, h)
        w = long_side
        h = long_side
    else:
        short_side = min(w, h)
        w = short_side
        h = short_side

    roi.set_width(w * x_scale)
    roi.set_height(h * y_scale)

    return roi


def TransformNormalizedRect(roi, input_width, input_height):
    '''referring to: the roi is normalized roi
    mediapipe/calculators/util/rect_transformation_calculator.cc
    '''
    square_long = True
    w = roi.width
    h = roi.height
    rot = roi.rotation
    rot = rot / math.pi * 180.

    x_shift = 0.
    y_shift = 0.
    x_scale = 1.
    y_scale = 1.

    if rot == 0.:
        roi.set_x(roi.center_x + w * x_shift)
        roi.set_y(roi.center_y + h * y_shift)
    else:
        x_shift = (input_width * w * x_shift * math.cos(rot) - input_height * h * y_shift * math.sin(rot)) / input_width
        y_shift = (input_width * w * x_shift * math.sin(rot) + input_height * h * y_shift * math.cos(
            rot)) / input_height
        roi.set_x(roi.center_x + x_shift)
        roi.set_y(roi.center_y + y_shift)

    if square_long:
        long_side = max(w * input_width, h * input_height)
        w = long_side / input_width
        h = long_side / input_height
    else:
        short_side = min(w * input_width, h * input_height)
        w = short_side / input_width
        h = short_side / input_height

    roi.set_width(w * x_scale)
    roi.set_height(h * y_scale)

    return roi


def detectionRemovalLetterbox(norm_point, letter_box):
    '''referring to :
    mediapipe/calculators/util/detection_letterbox_removal_calculator.cc'''
    left = letter_box[0]
    top = letter_box[1]
    left_and_right = letter_box[0] + letter_box[2]
    top_and_bottom = letter_box[1] + letter_box[3]
    x, y = norm_point
    new_x = (x - left) / (1.0 - left_and_right)
    new_y = (y - top) / (1.0 - top_and_bottom)
    return [new_x, new_y]


def landmarksRemovalLetterBox(norm_landmark, letter_box):
    '''referring to :
    mediapipe/calculators/util/landmark_letterbox_removal_calculator.cc'''
    left = letter_box[0]
    top = letter_box[1]
    left_and_right = letter_box[0] + letter_box[2]
    top_and_bottom = letter_box[1] + letter_box[3]
    x, y, z, presence, visibilty = norm_landmark
    new_x = (x - left) / (1.0 - left_and_right)
    new_y = (y - top) / (1.0 - top_and_bottom)
    new_z = z / (1.0 - left_and_right)  # Scale Z coordinate as X
    return [new_x, new_y, new_z, presence, visibilty]


def landmarkProjection(norm_landmark, roi):
    '''to project the norm landmark to original input scale'''
    angle_ = roi.rotation * math.pi / 180.0

    x, y, z, presence, visibilty = norm_landmark

    new_x = math.cos(angle_) * (x - 0.5) - math.sin(angle_) * (y - 0.5)
    new_y = math.sin(angle_) * (x - 0.5) + math.cos(angle_) * (y - 0.5)

    new_x = new_x * roi.width + roi.center_x
    new_y = new_y * roi.height + roi.center_y
    new_z = z * roi.width  # Scale Z coordinate as X.

    return [new_x, new_y, new_z, presence, visibilty]


def landmarkToDetection(norm_landmarks):
    '''to transform the normalized landmarks to the detection results'''

    xmin = np.min(norm_landmarks[:33, 0])
    xmax = np.max(norm_landmarks[:33, 0])
    ymin = np.min(norm_landmarks[:33, 1])
    ymax = np.max(norm_landmarks[:33, 1])

    return [xmin,ymin,xmax-xmin,ymax-ymin]

# r = ROI()
# r.set_value(0.5, 0.5, 1, 1, 0)
#
# new_r = getRoi(r, 256, 256, True)
#
# print(new_r.center_x, new_r.center_y)
# rect = np.array([[0,0],[0,1],[1,1],[1,0]])
# center = np.array([0.5,0.5])
#
# new_rect = rotate_points(rect,center,90)
# print(new_rect)
