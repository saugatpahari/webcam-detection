from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialize the model
model = YOLO("../yolo-weights/yolov8l.pt") # the path to your YOLO model file

# Open the video capture (0 for the first webcam, or replace with video file path)
data = cv2.VideoCapture(0)

# setting the width and height of webcam output view
data.set(3, 1280)
data.set(4, 720)

# Class Names for the objects detected by the webcam
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Error handling for video source
if not data.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Capture frame-by-frame
    success, video = data.read()

    # Error handling for frame read of video
    if not success:
        print("Error: Could not read frame.")
        break

    # Predict with YOLO model
    output = model(video, stream=True)

    # Visualize results on the frame
    for o in output:
        boxes = o.boxes
        # contains the bounding box information
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),  int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            # creating the rectangle around the object detected using both cvzone and cv2
            cvzone.cornerRect(video, (x1, y1, w, h), 25, 6, 3, (182, 249, 159), (27, 19, 37))
            # cv2.rectangle(video, (x1, y1), (x2, y2), (230, 230, 250), 4)

            # Getting the confidence level of the object detected
            confidence_value = math.ceil(box.conf[0]*100)/100

            # Getting the value of the object detected to associate with the classnames
            name = int(box.cls[0])

            # Displaying the captured objects names associated to the above classnames
            cvzone.putTextRect(video, f'{classNames[name]} {confidence_value}', (max(0, x1), max(30, y1-18)), scale=0.8, thickness=1, colorT=(0, 0, 0), colorR=(255, 255, 255))

    # Display the resulting frame
    cv2.imshow("Web Cam Video", video)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture and close windows
data.release()
cv2.destroyAllWindows()