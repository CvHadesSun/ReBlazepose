import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = ['./data/0.jpg']
cap = cv2.VideoCapture('./data/1_bodyweight_squats__tc__.webm')

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5) as pose:
  # for idx, file in enumerate(IMAGE_FILES):
  #   image = cv2.imread(file)
  while True:
    ret,image = cap.read()
    if ret:
      image_height, image_width, _ = image.shape
      # Convert the BGR image to RGB before processing.
      results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      if not results.pose_landmarks:
        continue
      # print(
      #     f'Nose coordinates: ('
      #     f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
      #     f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
      # )
      # Draw pose landmarks on the image.
      annotated_image = image.copy()
      mp_drawing.draw_landmarks(
          annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
      cv2.imshow('show',annotated_image)
    # cv2.waitKey()
    if cv2.waitKey(1)==ord('q'):break
    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)