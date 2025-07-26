#This file will use opencv to capture camera frames live
import cv2
import sys
import numpy as np

def capture_image():
    cap = cv2.VideoCapture('Taylor_Swift.mp4')  # Replace with your video file path or camera index
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'captured_image_{i}.jpg', rgb_image)  # Save the RGB image
        cv2.imshow('RGB Frame', frame)    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()


if __name__ == "__main__":
    capture_image()
    cv2.destroyAllWindows()
    
