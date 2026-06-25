Object Detection and Tracking System 

This is a real-time object detection and tracking system using a webcam or video input.
It uses a pre-trained YOLOv8 model for object detection and the SORT algorithm for tracking objects with unique IDs across frames.


1. Features
   
   1) Real-time webcam/video detection
   2) Object detection using YOLOv8
   3) Object tracking using SORT algorithm
   4) Bounding boxes with labels
   5) Unique tracking IDs for each object
   6) Lightweight and fast processing


2. Technologies Used
   
   1) Python
   2) YOLOv8 (Ultralytics) -> Object Detection
   3) SORT tracking algorithm -> Object Tracking
   4) OpenCV -> Video Processing
   5) NumPy


4. How to Run the Project
   
   1) Install dependencies using command:
      pip install -r requirements.txt
      
   3) Run the project using command:
      python yolo_sort_tracking.py
      
   5) To exit the program
      Press: Q

5. Project Output

   When you run the program:

   -Webcam window opens
   -Objects are detected in real-time
   -Bounding boxes are drawn around objects
   -Each object is labeled (person, bottle, phone, etc.)
   -Each object gets a unique tracking ID (ID 1, ID 2, etc.)
   -Objects are tracked across frames even when they move

7. Workflow of the System
   
   1) Webcam / Video Input
   2) Frame captured using OpenCV
   3) YOLOv8 detects objects in frame
   4) Bounding boxes + class labels generated
   5) SORT algorithm assigns unique tracking IDs
   6) Final output displayed in real-time
  

  Author:
  Sakshi Bondre
  

 
