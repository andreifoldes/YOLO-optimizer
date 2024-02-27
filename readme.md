This module contains functions for building and using YOLOv8n, YOLOv8m, and YOLOv8l face detection models.
It also includes functions for detecting faces in images using these models and evaluating the detection results.

Functions:
- build_model_yolov8n(): Builds and returns a YOLOv8n face detector model.
- build_model_yolov8m(): Builds and returns a YOLOv8m face detector model.
- build_model_yolov8l(): Builds and returns a YOLOv8l face detector model.
- detect_face(model_name, face_detector, img, confidence=0.95, align=False): Detects faces in an image using a face detector model.
- create_binary_vectors(df_truth, df_algo): Creates binary vectors for ground truth and algorithm results.

Note:
- The YOLOv8n, YOLOv8m, and YOLOv8l models are initialized using pre-trained weights downloaded from specified URLs.
- The detect_face function returns a list of tuples containing information about the detected faces in an image.
- The create_binary_vectors function is used for preparing the data for evaluation of face detection results.