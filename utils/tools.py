#@author:cvhadessun
#date:2021-5-12

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import copy
import math
# def visJoints(image,joints,save_path):
#     h,w,_ = image.shape


def draw2DJoint(image,joints):
    img=copy.copy(image)
    # w= 1080
    # h = 1920
    # print(joints)

    # img = cv2.resize(image,(w,h))
    joints[:,0] = joints[:,0] * image.shape[1]
    joints[:, 1] = joints[:, 1]  * image.shape[0]
    # joints[:, 0] = joints[:, 0]
    # joints[:, 1] = joints[:, 1]

    for i in range(joints.shape[0]):
        x,y=joints[i]
        if x>0 and y>0:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    # print(img.shape)
    # cv2.imwrite('./test1.jpg', img)
    # cv2.imshow('show',img)
    # cv2.waitKey()
    return img




def visJoints(m_data):
    # color = ''
    colors = '#DC143C'
    # colors3 = '#000000'
    # for i in range(len(cam_data)):
    #     if i == index:
    #         color = '#DC143C'
    #     else:
    #         color = '#000000'

        #c
        # _=comparaJoint(m_data,cam_data[i],color)
    area = np.pi * 4 ** 2
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(m_data[:, 0], m_data[:, 1], s=area, c=colors, alpha=0.4, label='m')
    plt.show()
    return True

def plot_detections(img, detections, with_keypoints=True):
    plt.gca().set_aspect('equal', adjustable='box')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.grid(False)
    ax.imshow(img)
    keypoints=[]


    print(img.shape)
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

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

