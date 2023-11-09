import cv2
import openvino as ov

from Detector import Detector
from LandmarkDetector import LandmarkDetector
from MusicPlayer import play_alarm_sound

# use face-detection-adas-0001 to detect face
# use facial-landmarks-98-detection to detect facial landmarks (eyes)
# use open-closed-eye to detect if eyes are open or closed

OPEN_CLOSED_THRESHOLD = 0.5
FACE_DETECTION_THRESHOLD = 0.5
EYES_CLOSED_COUNTER_THRESHOLD = 5
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

    cv2.putText(frame, 'Press ESC to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    face_detect_result = face_detector.detect(frame)

    '''
    Inputs
    Image, name: data, shape: 1, 3, 384, 672 in the format B, C, H, W, where:
    
    B - batch size
    C - number of channels
    H - image height
    W - image width
    Expected color order is BGR.
    
    Outputs The net outputs blob with shape: 1, 1, 200, 7 in the format 1, 1, N, 7, where N is the number of detected 
    bounding boxes. The results are sorted by confidence in decreasing order. Each detection has the format 
    [image_id, label, conf, x_min, y_min, x_max, y_max], where:
    
    image_id - ID of the image in the batch
    label - predicted class ID (1 - face)
    conf - confidence for the predicted class
    (x_min, y_min) - coordinates of the top left bounding box corner
    (x_max, y_max) - coordinates of the bottom right bounding box corner
    '''

    # filter out face detection results with confidence(detection[2]) < 0.5
    valid_detections = [detection for detection in face_detect_result[0][0] if detection[2] > FACE_DETECTION_THRESHOLD]
    # frame shape: height, width, channels. get height and width
    frame_h, frame_w = frame.shape[:2]

    if len(valid_detections) == 0:
        cv2.putText(frame, 'No Face Detected', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        continue

    for detection in valid_detections:
        image_id, label, conf, x_min, y_min, x_max, y_max = detection

        x_min = int(x_min * frame_w)
        y_min = int(y_min * frame_h)
        x_max = int(x_max * frame_w)
        y_max = int(y_max * frame_h)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # crop face
        face = frame[y_min:y_max, x_min:x_max]

        # detect facial landmarks
        landmark_detect_result = landmark_detector.detect(face)
        left_eye, right_eye = landmark_detector.extract_eyes_from_output(face, landmark_detect_result)

        left_eye_detect_result = open_closed_eye_detector.detect(left_eye)
        right_eye_detect_result = open_closed_eye_detector.detect(right_eye)

        left_eye_open_prob = left_eye_detect_result[0][1][0][0]
        right_eye_open_prob = right_eye_detect_result[0][1][0][0]

        if left_eye_open_prob < OPEN_CLOSED_THRESHOLD and right_eye_open_prob < OPEN_CLOSED_THRESHOLD:
            cv2.putText(frame, 'Eyes Closed', (x_min, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 0, 0), 2)
            EYES_CLOSED_COUNTER += 1
        else:
            cv2.putText(frame, 'Eyes Open', (x_min, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            EYES_CLOSED_COUNTER = 0

        if EYES_CLOSED_COUNTER > EYES_CLOSED_COUNTER_THRESHOLD:
            EYES_CLOSED_COUNTER = 0
            play_alarm_sound('beep.mp3')

        cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
