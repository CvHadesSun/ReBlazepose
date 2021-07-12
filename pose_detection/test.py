# [tensor([[-0.0173,  0.4400,  0.3306,  0.7879,  0.5985,  0.4946,  0.6157, -0.0605,
#           0.6303,  0.2233,  0.6426, -0.1343,  0.6387]])]
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import cv2
import math
from pose_detection.blaze_detection import   main

img_dir = './0.jpg'
detections=main(img_dir,'./pose_detection.tflite','./anchors.npy')
outputs = detections[0].numpy()

# outputs = np.array([[-0.0173,  0.4400,  0.3306,  0.7879,  0.5985,  0.4946,  0.6157, -0.0605,
#           0.6303,  0.2233,  0.6426, -0.1343,  0.6387]])

# outputs = outputs[0]
img=cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(outputs)

def plot_detections(img, detections, with_keypoints=True):
    plt.gca().set_aspect('equal', adjustable='box')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    keypoints=[]
    # if isinstance(detections, torch.Tensor):
    #     detections = detections.cpu().numpy()
    #
    # if detections.ndim == 1:
    #     detections = np.expand_dims(detections, axis=0)

    # print("Found %d faces" % detections.shape[0])

    print(img.shape)
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]
        # x_center = detections[i,0] *img.shape[1]
        # y_center = detections[i,1] *img.shape[0]
        # w = detections[i,2] *img.shape[1]
        # h = detections[i,3] *img.shape[0]
        # xmin = x_center - w/2
        # ymin = y_center - h/2
        # xmax = x_center + w/2
        # ymax = y_center + h/2

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none",
                                 alpha=detections[i, 12])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(4):
                kp_x = detections[i, 4 + k * 2] * img.shape[1]
                kp_y = detections[i, 4 + k * 2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=5, linewidth=5,
                                        edgecolor="red", facecolor="none",
                                        alpha=detections[i, 12])
                keypoints.append([kp_x,kp_y])
                ax.add_patch(circle)

        radius_ = math.sqrt((keypoints[0][0]-keypoints[1][0])**2+(keypoints[0][1]-keypoints[1][1])**2)

        circle = patches.Circle((keypoints[0][0],keypoints[0][1]), radius=radius_, linewidth=5,
                                edgecolor="blue", facecolor="none",
                            alpha=1)
        ax.add_patch(circle)
    # print(keypoints)
    #     print((detections[i,-1]))
    plt.show()

plot_detections(img,outputs)
# # img = cv2.resize(img,(128,128))
# outputs=outputs[0]
# h,w,_ = img.shape
# ymin = outputs[0]*h
# xmin = outputs[1]*w
# ymax = outputs[2]*h
# xmax = outputs[3]*w
# # # ymin = 0
# hip_x = outputs[4]*w
# hip_y = outputs[5]*h
# # #
# up_body_x = outputs[6]*w
# up_body_y = outputs[7]*h
# #
# # #
# shoulder_y = outputs[9]*h
# shoulder_x = outputs[8]*w
# # #
# down_body_x = outputs[10]*w
# down_body_y = outputs[11]*h
# #
# print(w,h)
# print(up_body_x,up_body_y)
# print(down_body_x,down_body_y)
#
# cv2.circle(img, (int(xmin), int(ymin)), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
# cv2.circle(img, (int(xmin), int(ymax)), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
# cv2.circle(img, (int(xmax), int(ymin)), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
# cv2.circle(img, (int(xmax), int(ymax)), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
#
# cv2.circle(img, (int(hip_x), int(hip_y)), 5, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
# cv2.circle(img, (int(shoulder_x), int(shoulder_y)), 5, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
#
# cv2.circle(img, (int(up_body_x), int(up_body_y)), 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
# cv2.circle(img, (int(down_body_x), int(down_body_y)), 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
#
#
# cv2.imshow('show',img)
# cv2.waitKey()
# print(h,w)