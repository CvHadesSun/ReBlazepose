#author:cvhadessun
#date:2021.6.1

'''to implement the image processor of mediapipe(image_to_tensor_calculator fun)'''
import numpy as np
import math
import cv2

# Transforms the input image into a 128x128 while keeping the aspect ratio
# (what is expected by the corresponding model), resulting in potential
# letterboxing in the transformed image.
# node: {
#   calculator: "ImageToTensorCalculator"
#   input_stream: "IMAGE_GPU:image"
#   output_stream: "TENSORS:input_tensors"
#   output_stream: "LETTERBOX_PADDING:letterbox_padding"
#   options: {
#     [mediapipe.ImageToTensorCalculatorOptions.ext] {
#       output_tensor_width: 128
#       output_tensor_height: 128
#       keep_aspect_ratio: true
#       output_tensor_float_range {
#         min: -1.0
#         max: 1.0
#       }
#       border_mode: BORDER_ZERO
#       gpu_origin: TOP_LEFT
#     }
#   }
# }

# class ROI(object):
#     '''data struct for ROI
#         @x:center x of roi box
#         @y:center y of roi box
#         @w:
#         @h:
#         @rotation: angle in degree
#         '''
#     def __init__(self,x,y,w,h,rotation):
#         self.center_x= x
#         self.center_y = y
#         self.width=w
#         self.height=h
#         self.rotation=rotation


def ImageGetPadding(output_w,output_h,roi):
    '''
    use output size and roi size to get the padding parmeters in normalization,
    @return:padding [horizontal_padding,vertical_padding,horizontal_padding,vertical_padding],
    vertical_padding = (w'-h)/2/w' ;and  horizontal_padding = (h'-w)/2/h'
    roi:update the w and h only.
    '''


    output_ratio = output_h/output_w

    roi_ratio= roi.height/roi.width
    new_width=0
    new_height = 0
    vertical_padding=0.0
    horizontal_padding =0.0
    if output_ratio>roi_ratio:
        new_width =roi.width
        new_height = roi.width*output_ratio
        vertical_padding = (1.- roi_ratio/output_ratio)/2.
    else:
        new_width = roi.height/output_ratio
        new_height =roi.height
        horizontal_padding = (1.-output_ratio/roi_ratio)/2.

    roi.width = new_width
    roi.height = new_height

    padding = [horizontal_padding,vertical_padding,horizontal_padding,vertical_padding]
    return padding, roi

def GetRotatedSubRectToRectTransformMatrix(roi,image_width,image_height,flip=False):
    '''to get transform matrix'''

    #  Matrix to convert X,Y to [-0.5, 0.5] range "initial_translate_matrix"
    # { 1.0f,  0.0f, 0.0f, -0.5f}
    # { 0.0f,  1.0f, 0.0f, -0.5f}
    # { 0.0f,  0.0f, 1.0f,  0.0f}
    # { 0.0f,  0.0f, 0.0f,  1.0f}
    a= roi.width
    b=roi.height
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

    c= math.cos(roi.rotation/180 *math.pi)
    d= math.sin(roi.rotation/180 *math.pi)
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

    g = 1./image_width
    h = 1./image_height
    # Matrix to scale X,Y,Z to [0.0, 1.0] range "post_scale_matrix"
    # {g,    0.0f, 0.0f, 0.0f}
    # {0.0f, h,    0.0f, 0.0f}
    # {0.0f, 0.0f,    g, 0.0f}
    # {0.0f, 0.0f, 0.0f, 1.0f}

    #matrix

    #row 1
    mat = np.ones([4,4],dtype=np.float)
    mat[0,0] = a * c * fl * g
    mat[0,1] = -b * d * g
    mat[0,2] = 0.
    mat[0,3] =(-0.5 * a * c * fl + 0.5 * b * d + e) * g
    #row 2
    mat[1,0] = a * d * fl * h
    mat[1,1] =  b * c * h
    mat[1,2] = 0.
    mat[1.3] = (-0.5 * b * c - 0.5 * a * d * fl + f) * h
    #row 3
    mat[2,0] = 0.
    mat[2,1] = 0.
    mat[2,2] = a*g
    mat[2,3] = 0.
    #row 4
    mat[3,0] = 0.
    mat[3,1] = 0.
    mat[3,2] = 0.
    mat[3,3] = 1.

    return mat







