import landmark_smooth
from landmark_smooth.landmark_smooth_calculator import GetObjectScaleFromNormROI


def FilterAuxLandmarks(vis_filter, landmark_filter, aux_landmarks, timestamp, input_width, input_height, norm_roi):
    '''filter the aux_landmarks'''
    # print(aux_landmarks.shape)
    object_scale = GetObjectScaleFromNormROI(norm_roi, input_width, input_height)
    # transform normalized landmarks to origin scale
    aux_landmarks[:, 0] *= input_width
    aux_landmarks[:, 1] *= input_height
    aux_landmarks[:, 2] *= input_width
    # print(aux_landmarks)
    filtered_vis_landmarks = vis_filter.Apply(aux_landmarks, timestamp)
    filtered_aux_landmarks = landmark_filter.Apply(filtered_vis_landmarks, timestamp, object_scale)

    # nomalize the filtered landmarks
    filtered_aux_landmarks[:, 0] /= input_width
    filtered_aux_landmarks[:, 1] /= input_height
    filtered_aux_landmarks[:, 2] /= input_width

    return filtered_aux_landmarks, vis_filter, landmark_filter


def FitlerLandmarks(vis_filter, landmark_filter, landmarks, timestamp, input_width, input_height,
                    norm_roi):
    object_scale = GetObjectScaleFromNormROI(norm_roi, input_width, input_height)
    # transform normalized landmarks to origin scale
    landmarks[:, 0] *= input_width
    landmarks[:, 1] *= input_height
    landmarks[:, 2] *= input_width
    filtered_landmarks = vis_filter.Apply(landmarks, timestamp)
    filtered_landmarks = landmark_filter.Apply(filtered_landmarks, timestamp, object_scale)

    # nomalize the filtered landmarks
    filtered_landmarks[:, 0] /= input_width
    filtered_landmarks[:, 1] /= input_height
    filtered_landmarks[:, 2] /= input_width

    return filtered_landmarks, vis_filter, landmark_filter


def initVisFilter(filter_name, params):
    return eval('landmark_smooth.' + filter_name)(params)


def initOneEuroFilter(filter_name, params):
    return eval('landmark_smooth.' + filter_name) \
        (params[0],
         params[1],
         params[2],
         params[3],
         params[4],
         params[5])
