import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid   # Unique identifier
import os
import time
import pandas as pd
from PIL import Image

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', source='github', force_reload=True)

# Define the video file to read from
filename = "IMG420"
cap = cv2.VideoCapture(f"test/{filename}.MOV")

# Create an empty dataframe to store the object detection results
df_prediction = pd.DataFrame()

# Start reading frames from the video
frame_num = -1
while cap.isOpened():
    frame_num += 1
    ret, frame = cap.read()

    # If we have reached the end of the video, break out of the loop
    if not ret:
        break

    # Run the YOLOv5 model on the current frame
    results = model(frame)

    # Extract the bounding box coordinates and confidence scores for each detected object
    bboxes = results.xyxy[0].numpy()[:, :-1]  # remove last column (confidence)

    # Create a dataframe to store the detection results for this frame
    temp_df = pd.DataFrame(bboxes, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence'])
    temp_df['frame'] = frame_num


    # Append the detection results to the overall dataframe
    df_prediction = df_prediction.append(temp_df)

    # Display the frame with the bounding boxes drawn around the detected objects
    img = np.squeeze(results.render())
    cv2.imshow('YOLO', img)

    # Save the frame as a JPEG file
    cv2.imwrite(f"img{frame_num}.jpg", img)

    # Wait for a key press to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
