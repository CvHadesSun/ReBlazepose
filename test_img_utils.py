import cv2
import os
from utils import img_process_tools
from config.cfg import cfg
from pose_detection.model import PoseDetection
from utils.tools import plot_detections

root_path = os.path.abspath(os.path.join('.'))
cfg['root_dir'] = root_path

img=cv2.imread('./data/beauity_motion.jpg')

# pd = PoseDetection(cfg)
#
# roi = pd(img)


# print(coord)
#
# plot_detections(img,coord[0])
h,w,_=img.shape
#
roi = img_process_tools.ROI(w/2,h/2,w/4,h/4,90)
# roi.set_value(w/2,h/2,w,h,-90)
cv2.rectangle(img, (int(w/2-w/4/2), int(h/2-h/4/2)), (int(w/2+w/4/2), int(h/2+h/4/2)), (0, 255, 0), 2)

roi.set_x(roi.center_x/w)
roi.set_y(roi.center_y/h)
roi.set_width(roi.width/w)
roi.set_height(roi.height/h)

# new_roi = img_process_tools.TransformRect(roi)
new_roi=img_process_tools.TransformNormalizedRect(roi,w,h)
#
new_roi.set_x(roi.center_x*w)
new_roi.set_y(roi.center_y*h)
new_roi.set_width(roi.width*w)
new_roi.set_height(roi.height*h)


cv2.rectangle(img, (int(new_roi.center_x-new_roi.width/2),
                    int(new_roi.center_y-new_roi.height/2)),
              (int(new_roi.center_x+new_roi.width/2),
               int(new_roi.center_y+new_roi.height/2)),
              (255, 0, 0), 2)

# # new_roi=img_process_tools.getRoi(roi,w,h,False)
# new_img = img_process_tools.ImageTransform(img,roi,[256,256],[0,1])
#
# for i in range(rot_points.shape[0]):
#     x,y = rot_points[i]
#     cv2.circle(img,(int(x),int(y)),10,(255,0,0),-1)

cv2.imshow('show',img)
cv2.waitKey()
