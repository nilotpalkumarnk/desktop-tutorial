import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('yolov3.h5')

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
           'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, (416, 416))

    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize the frame
    frame = frame.astype('float32') / 255.0

    # Add batch dimension
    frame = np.expand_dims(frame, axis=0)

    # Predict the classes and locations
    boxes, scores, classes = model.predict(frame)

    # Draw the boxes on the frame
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1 = int(x1 * frame.shape[2])
        y1 = int(y1 * frame.shape[1])
        x2 = int(x2 * frame.shape[2])
        y2 = int(y2 * frame.shape[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, classes[np.argmax(scores[i])], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Wait for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
