"""
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
"""

#%% setup

import os
from scipy.datasets import face
import gdown
# from deepface.detectors import FaceDetector
# from deepface.commons import functions

def build_model_yolov8n():
  """
  Builds and returns a YOLOv8n face detector model.

  This function checks if the YOLOv8n face detector model weights are already downloaded.
  If not, it downloads the weights from a specified URL and saves them in the appropriate directory.
  Once the weights are downloaded, it initializes the YOLOv8n face detector model using the downloaded weights.

  Returns:
  - face_detector: An instance of the YOLOv8n face detector model.

  Raises:
  - None

  Example usage:
  >>> face_detector = build_model_yolov8n()
  >>> print("Face detector model ready")
  """

  home = "YOLO-optimizer/weights"
  if os.path.isfile(home + "/.deepface/weights/yolov8n-face.pt") != True:
    url = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
    output = home + "/.deepface/weights/yolov8n-face.pt"
    # create the directory if it doesn't exist
    if not os.path.exists(home + "/.deepface/weights"):
      os.makedirs(home + "/.deepface/weights")
    gdown.download(url, output, quiet=False)
    print("Downloaded YOLO model yolo8vn-face.pt")
  
  from ultralytics import YOLO
  face_detector = YOLO(home + "/.deepface/weights/yolov8n-face.pt")
  print("YOLOv8n ready")
  
  return face_detector

# see https://github.com/derronqi/yolov7-face

def build_model_yolov8m():
  """
  Builds and returns a YOLOv8m face detection model.

  Returns:
    YOLO: The YOLOv8m face detection model.
  """
  home = "YOLO-optimizer/weights"
  if os.path.isfile(home + "/.deepface/weights/yolov8m-face.pt") != True:
    url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-face.pt"
    output = home + "/.deepface/weights/yolov8m-face.pt"
    # create the directory if it doesn't exist
    if not os.path.exists(home + "/.deepface/weights"):
      os.makedirs(home + "/.deepface/weights")
    # gdown.download(url, output, quiet=False)
    # download the model from the github release
    gdown.download(url, output, quiet=False)
    print("Downloaded YOLO model yolo8vm-face.pt")
  from ultralytics import YOLO
  face_detector = YOLO(home + "/.deepface/weights/yolov8m-face.pt")
  print("YOLOv8m ready")
  return face_detector

def build_model_yolov8l():
  """
  Builds and returns a YOLOv8l face detection model.

  This function checks if the YOLOv8l face detection model weights are already downloaded.
  If not, it downloads the weights from a GitHub release and saves them in the appropriate directory.
  Once the weights are available, it initializes the YOLOv8l model using the downloaded weights.

  Returns:
    face_detector (YOLO): The YOLOv8l face detection model.

  Raises:
    None
  """
  home = "YOLO-optimizer/weights"
  if os.path.isfile(home + "/.deepface/weights/yolov8l-face.pt") != True:
    url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt"
    output = home + "/.deepface/weights/yolov8l-face.pt"
    # create the directory if it doesn't exist
    if not os.path.exists(home + "/.deepface/weights"):
      os.makedirs(home + "/.deepface/weights")
    # gdown.download(url, output, quiet=False)
    # download the model from the github release
    gdown.download(url, output, quiet=False)
    print("Downloaded YOLO model yolo8vl-face.pt")
  from ultralytics import YOLO
  face_detector = YOLO(home + "/.deepface/weights/yolov8l-face.pt")
  print("YOLOv8l ready")
  return face_detector

import cv2

def detect_face(model_name, face_detector, img, confidence=0.95, align=False):
  """
  Detects faces in an image using a face detector model.

  Args:
    model_name (str): The name of the face detector model.
    face_detector: The face detector model object.
    img (str): The path to the image file.
    confidence (float, optional): The confidence threshold for face detection. Defaults to 0.95.
    align (bool, optional): Whether to align the detected faces. Defaults to False.

  Returns:
    list: A list of tuples containing information about the detected faces.
      Each tuple contains the image name, face index (0 IF no face detected OR [1...n] if more than zero faces detected, depending on the index), detected face image, region coordinates, confidence, and model name.
      
  Note:
    The region coordinates are in the format (x, y, w, h), where x and y are the top-left corner of the detected face, and w and h are the width and height of the detected face, respectively.
    If there are e.g.: 3 faces detected in the image, the function should return 3 tuples, each containing information about the detected face.
  """
  resp = []
  align=False
  detected_face = None
  detections = [tuple(int(x) for x in sublist) for sublist in face_detector(img, verbose=False, conf=confidence)[0].boxes.xywh.tolist()]

  img_loaded = cv2.imread(img)

  if len(detections) > 0:
    
    for i, detection in enumerate(detections):
      x,y,w,h = detection[0], detection[1], detection[2], detection[3]
      img_region = [x, y, w, h]
      # Crop the image to the detected face
      detected_face = img_loaded[int(y) : int(y + h), int(x) : int(x + w)]
      # resp.append((img, i+1, detected_face, img_region, confidence))
      # get basename of the image
      resp.append((os.path.basename(img), i+1, img_region, confidence, model_name))
  else:
      resp.append((os.path.basename(img), 0, None, confidence, model_name))

  return resp

#%% choose the model

# test the model
face_detector = build_model_yolov8n()
face_detector = build_model_yolov8m()
face_detector = build_model_yolov8l()

#%% test with simple grid search

from sklearn.model_selection import ParameterGrid
# Define the parameter grid
param_grid = {'confidence': [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.8,0.9]}
# Create the grid
grid = ParameterGrid(param_grid)
# download test image of a face
# face_detector = build_model_yolov8l()

# create a dictionary of different face detection models
face_detectors = {
    'yolov8n': build_model_yolov8n(),
    'yolov8m': build_model_yolov8m(),
    'yolov8l': build_model_yolov8l()
}

import glob
from tqdm import tqdm

def apply_face_detection(image_directory, face_detectors, grid):
  """
  Apply face detection on a directory of images using multiple face detectors and confidence levels.

  Args:
    image_directory (str): The directory path containing the images to be processed.
    face_detectors (dict): A dictionary of face detectors, where the keys are the model names and the values are the face detector objects.
    grid (list): A list of dictionaries, where each dictionary contains the parameters for testing, including the confidence level.

  Returns:
    list: A list of detections for each image, where each detection is a dictionary containing the model name, confidence level, and the detected faces.
  """
  total_detections = []
  png_files = glob.glob(f"{image_directory}/*.png")

  for model_name, face_detector in face_detectors.items():
    print(f"Testing with model: {model_name}")

    for params in grid:
      confidence = params['confidence']
      print(f"Testing with confidence: {confidence}")

      for image_file in tqdm(png_files):
        image_path = os.path.join(image_directory, image_file)
        detections = detect_face(model_name, face_detector, image_path, confidence=confidence)
        total_detections.append(detections)

  return total_detections


"""
  the heart of the code
    @@@       @@@
   @@@@@@   @@@@@@
  @@@@@@@@@@@@@@@@
   @@@@@@@@@@@@@@@
    @@@@@@@@@@@
      @@@@@@@@
       @@@@@
        @@@
          @
"""

# Usage example:
image_directory = "/home/xxx/scratch/object detection/Frames"
total_detections = apply_face_detection(image_directory, face_detectors, grid)

#% quick summary

# Flatten the sublists
flattened_detections = [sublist[0] for sublist in total_detections]
import pandas as pd
# Convert to DataFrame
df = pd.DataFrame(flattened_detections, columns=['path', 'face_detected', 'image_region', 'confidence_setting','name'])

# save the dataframe to a csv file
df.to_csv('YOLO-optimizer/face_detection_results.csv', index=False)

# calculate the number of faces detected for each confidence setting and model by counting non-zero face_detected values
df.groupby(['name', 'confidence_setting']).face_detected.apply(lambda x: (x != 0).sum()).reset_index()
# %% Ground truth

# load csv
import pandas as pd
df = pd.read_csv('YOLO-optimizer/face_detection_results.csv')

# load or construct ground truth dataframe
# for demo purposes we will designate model yolov8m, confidence 0.7 as the ground truth
df_ground_truth = df[(df['name'] == 'yolov8n') & (df['confidence_setting'] == 0.7)]

# validate ground truth dataframe

def check_ground_truth(df):
  """
  Check the validity of the ground truth DataFrame.

  Parameters:
  df (pandas.DataFrame): The DataFrame to be checked.

  Raises:
  ValueError: If the DataFrame does not have 'path' and 'face_detected' columns,
        if the DataFrame contains duplicate rows,
        or if the 'face_detected' column contains non-integer values.

  Returns:
  None
  """
  # Check if DataFrame has two columns named 'filename' and 'face_index'
  # filter only two columns of df
  df = df[['path', 'face_detected']]
  if 'path' not in df.columns or 'face_detected' not in df.columns:
    raise ValueError("DataFrame should have 'path' and 'face_detected' columns.")
  
  # Check if each row is unique
  if df.duplicated().any():
    # print(df.duplicated())
    print(df.duplicated())
    raise ValueError("Each row in the DataFrame should be unique.")
  
  # Check if 'face_index' only contains integers
  if not df['face_detected'].apply(lambda x: isinstance(x, int)).all():
    raise ValueError("'face_detected' should only contain integers.")
  else:
    print("Ground truth DataFrame is valid.")

# check the ground truth dataframe
check_ground_truth(df_ground_truth)
# check the number of the faces detected, by counting the number of non-zero face_detected values
# count the number of paths for each max(face_detected) value
df_ground_truth.groupby('face_detected').path.nunique()

#%% Preparing the data for evaluation

import pandas as pd

# def create_binary_vectors(df_truth, df_algo, n):
#     """
#     Create binary vectors for ground truth and algorithm results.
    
#     Parameters:
#     - df_truth: DataFrame with columns 'path' and 'face_detected' representing ground truth.
#     - df_algo: DataFrame with the same columns representing face detection algorithm results.
#     - n: The maximum number of possible face detections for any given image.

#     Returns:
#     - truth_vector: Binary vector for ground truth.
#     - algo_vector: Binary vector for algorithm results.
#     """
#     # Initialize binary vectors with zeros
#     truth_vector = pd.Series(0, index=pd.MultiIndex.from_product([df_truth['path'].unique(), range(n)]))
#     algo_vector = pd.Series(0, index=pd.MultiIndex.from_product([df_algo['path'].unique(), range(n)]))

#     # Fill the binary vectors with detections
#     for _, row in df_truth.iterrows():
#         truth_vector[(row['path'], row['face_detected'])] = 1
    
#     for _, row in df_algo.iterrows():
#         algo_vector[(row['path'], row['face_detected'])] = 1

#     # Align indices of the two vectors to ensure they have the same length
#     all_image_ids = sorted(set(truth_vector.index.get_level_values(0)).union(algo_vector.index.get_level_values(0)))
#     all_face_indices = range(n)
#     common_index = pd.MultiIndex.from_product([all_image_ids, all_face_indices])

#     truth_vector = truth_vector.reindex(common_index, fill_value=0)
#     algo_vector = algo_vector.reindex(common_index, fill_value=0)

#     return truth_vector, algo_vector

def create_binary_vectors(df_truth, df_algo):
    # Group by 'path' and check if at least one face was detected for each 'path'
    truth_vector = (df_truth.groupby('path')['face_detected'].sum() > 0).astype(int)
    algo_vector = (df_algo.groupby('path')['face_detected'].sum() > 0).astype(int)

    # Align the vectors
    truth_vector, algo_vector = truth_vector.align(algo_vector, join='outer')

    # Fill NaN values with 0 (no face detected)
    truth_vector.fillna(0, inplace=True)
    algo_vector.fillna(0, inplace=True)

    return truth_vector, algo_vector


# Example usage with dummy data:
# lets assume we have 3 images, where in truth only the second image has a face detected, and in algo the first and third images have a face detected, where in the first image there are two faces detected
df_truth = pd.DataFrame({'path': ['img_001', 'img_002', 'img_003', 'img_004'], 'face_detected': [1, 1, 0,0]})
df_algo = pd.DataFrame({'path': ['img_001', 'img_001', 'img_002' ,'img_003', 'img_004'], 'face_detected': [1, 2, 0, 1,1]})

# Max possible detection (including 0)
n = max(df_truth['face_detected'].max(), df_algo['face_detected'].max())+1

truth_vector, algo_vector = create_binary_vectors(df_truth, df_algo)

# convert series to numpy array
truth_vector_np = truth_vector.to_numpy()
algo_vector_np = algo_vector.to_numpy()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
  """
  Plots the confusion matrix for the given true labels (y_true) and predicted labels (y_pred).

  Args:
    y_true: Ground truth labels (1D array).
    y_pred: Predicted labels from an algorithm (1D array).

  Returns:
    None
  """
  # Calculate the confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  # Plot the confusion matrix
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title("Confusion Matrix")
  plt.show()

# Example usage:
plot_confusion_matrix(truth_vector_np, algo_vector_np)

from sklearn.metrics import precision_recall_fscore_support
def calculate_metrics(y_true, y_pred):
  """
  Calculates precision, recall, F1-score, and miss rate for a given y_true and y_pred pair.

  Args:
      y_true: Ground truth labels (1D array).
      y_pred: Predicted labels from an algorithm (1D array).

  Returns:
      precision: Precision score.
      recall: Recall score.
      f1_score: F1-score.
      miss_rate: Miss rate.
  """
  precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred)

  # Miss rate for the "face" class (assuming class index 0)
  # miss_rate = 1 - recall[0]

  return precision[1], recall[1], f1_score[1], support[1]

# calculate the metrics
precision, recall, f1_score, support = calculate_metrics(truth_vector_np, algo_vector_np)

# print the metrics in a nice format, in one line
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}, Support: {support:.2f}")
# %% evaluate the results

# load or construct ground truth dataframe
# for demo purposes we will designate model yolov8n, confidence 0.7 as the ground truth
# create empty dataframe for results
df_results = pd.DataFrame(columns=['name', 'confidence_setting', 'precision', 'recall', 'f1_score', 'support'])
df_ground_truth = df[(df['name'] == 'yolov8m') & (df['confidence_setting'] == 0.8)]

# how many rows in df_ground_truth have non-zero face_detected values
df_ground_truth['face_detected'].astype(bool).sum(axis=0)

confidence_values = [d for d in df['confidence_setting'].unique()]
# iterate through all unique confidence settings and models
for model_name in df.name.unique():
    for confidence in confidence_values:
        # filter the dataframe for the model and confidence setting
        df_model_algo = df[(df['name'] == model_name) & (df['confidence_setting'] == confidence)]
        # sort the dataframe by path
        df_model_algo = df_model_algo.sort_values('path')
        # Max possible detection (including 0)
        # n = max(df_ground_truth['face_detected'].max(), df_model_algo['face_detected'].max())+1
        # create the binary vectors
        truth_vector, algo_vector = create_binary_vectors(df_ground_truth, df_model_algo)
        
        df_ground_truth['face_detected'].astype(bool).sum(axis=0)
        df_model_algo['face_detected'].astype(bool).sum(axis=0)

        # convert series to numpy array
        truth_vector_np = truth_vector.to_numpy()
        algo_vector_np = algo_vector.to_numpy()
        
        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(truth_vector_np, algo_vector_np)
        # print(cm)

        # calculate the metrics
        precision, recall, f1_score, support = calculate_metrics(truth_vector_np, algo_vector_np)
        # store the metrics in dataframe using pd.concat
        df_results = pd.concat([df_results, pd.DataFrame({'name': [model_name], 'confidence_setting': [confidence], 'precision': [precision], 'recall': [recall], 'f1_score': [f1_score], 'support': [support]})])

# arrange df_results by f1_score
df_results.sort_values(by='f1_score', ascending=False)

# %%

def pr_curve_plot(df):
    """
    Args:
        df (pandas.DataFrame): DataFrame with columns 'name', 'confidence_setting', 'precision', 'recall'
    """
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay
    
    _, ax = plt.subplots(figsize=(8, 6))
    plt.axhline(1,0,0.955, linestyle='--', label='Ideal')
    plt.axvline(1,0,0.955, linestyle='--')
    plt.title('Precision-Recall Curve')
    plt.plot()
    
    # Get the unique names of the algorithms
    names = df['name'].unique()

    # For each algorithm
    for name in names:
        # Filter the DataFrame to only include rows for this algorithm
        df_filtered = df[df['name'] == name]
        
        # Sort the DataFrame by confidence_setting
        df_sorted = df_filtered.sort_values('confidence_setting')
        
        # Plot the precision-recall curve
        PrecisionRecallDisplay(df_sorted['recall'], df_sorted['precision']).plot(ax=ax, name=name)

# filter row in df_results where f1_score is greater 1.00
df_results_filtered = df_results[df_results['f1_score'] < 0.999]
df_results_filtered = df_results_filtered[df_results_filtered['f1_score'] > 0.00]
# sort the dataframe by f1_score
df_results_filtered = df_results_filtered.sort_values(by='f1_score', ascending=False)
# Call the function
pr_curve_plot(df_results_filtered)

# %%
