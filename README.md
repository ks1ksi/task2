# task2

Intel Open Model Zoo의 pretrained model들을 사용하여 졸음 방지 Application을 구현했습니다.

## 1. Environment
- OpenVINO 2023.2.0.dev20230922
- Python 3.11.4

## 2. Model
- face-detection-adas-0001
  - 화면에서 얼굴 이미지를 추출하기 위해 사용했습니다.
- facial-landmarks-35-adas-0002
  - 얼굴 이미지에서 눈의 위치를 찾기 위해 사용했습니다.
- open-closed-eye-0001
  - 눈이 떠있는지 닫혀있는지 판단하기 위해 사용했습니다.

## 3. How to run
> pip install -r requirements.txt

> python main.py


## 4. Result

일정 시간동안 눈을 감고 있으면 경고음이 울리고, 눈을 뜨면 경고음이 멈춥니다. 