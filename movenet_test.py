import cv2
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path='./models/lite-model_movenet_singlepose_thunder_3.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(output_details)