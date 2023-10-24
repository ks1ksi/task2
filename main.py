import cv2
import openvino as ov

from Detector import Detector
from LandmarkDetector import LandmarkDetector

# use face-detection-adas-0001 to detect face
# use facial-landmarks-98-detection to detect facial landmarks (eyes)
# use open-closed-eye to detect if eyes are open or closed

THRESHOLD = 0.5
EYES_CLOSED_COUNTER = 0

core = ov.Core()

face_detection_model = core.read_model('intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml')
face_detection_model_compiled = core.compile_model(face_detection_model, 'AUTO')
face_detector = Detector(face_detection_model_compiled)

facial_landmarks_model = core.read_model(
    'intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml')
facial_landmarks_model_compiled = core.compile_model(facial_landmarks_model, 'AUTO')
landmark_detector = LandmarkDetector(facial_landmarks_model_compiled)

# 변환한 모델 사용
open_closed_eye_model = core.read_model('public/open-closed-eye-0001/FP32/open-closed-eye-0001.xml')
open_closed_eye_model_compiled = core.compile_model(open_closed_eye_model, 'AUTO')
open_closed_eye_detector = Detector(open_closed_eye_model_compiled)

print('Models loaded')
print(f"face_detection_model: {face_detection_model}")
print(f"facial_landmarks_model: {facial_landmarks_model}")
print(f"open_closed_eye_model: {open_closed_eye_model}")

# load video from laptop camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # press esc to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

    face_detect_result = face_detector.detect(frame)
    valid_detections = [detection for detection in face_detect_result[0][0] if detection[2] > THRESHOLD]
    frame_h, frame_w = frame.shape[:2]
    for detection in valid_detections:
        image_id, label, conf, x_min, y_min, x_max, y_max = detection

        x_min = int(x_min * frame_w)
        y_min = int(y_min * frame_h)
        x_max = int(x_max * frame_w)
        y_max = int(y_max * frame_h)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0),
                      2)  # BGR color for the box and thickness=2

        # crop face
        face = frame[y_min:y_max, x_min:x_max]

        # detect facial landmarks
        landmark_detect_result = landmark_detector.detect(face)

        left_eye, right_eye = landmark_detector.extract_eyes_from_output(face, landmark_detect_result)

        # cv2.imshow('left_eye', left_eye)
        # cv2.imshow('right_eye', right_eye)
        # cv2.moveWindow('left_eye', 700, 0)
        # cv2.moveWindow('right_eye', 1000, 0)

        left_eye_detect_result = open_closed_eye_detector.detect(left_eye)
        right_eye_detect_result = open_closed_eye_detector.detect(right_eye)

        left_eye_open_prob = left_eye_detect_result[0][1][0][0]
        right_eye_open_prob = right_eye_detect_result[0][1][0][0]

        print(f"left_eye_open_prob: {left_eye_open_prob}")
        print(f"right_eye_open_prob: {right_eye_open_prob}")

        if left_eye_open_prob < THRESHOLD and right_eye_open_prob < THRESHOLD:
            cv2.putText(frame, 'Eyes Closed', (x_min, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 0, 0), 2)
            EYES_CLOSED_COUNTER += 1
        else:
            cv2.putText(frame, 'Eyes Open', (x_min, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            EYES_CLOSED_COUNTER = 0

        if EYES_CLOSED_COUNTER > 10:
            cv2.putText(frame, 'Drowsiness Detected', (x_min, y_min + 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (0, 0, 255), 3)
            EYES_CLOSED_COUNTER = 0

        cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
