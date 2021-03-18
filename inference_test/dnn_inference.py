import time
import cv2
import numpy as np
from pathlib import Path


PATH_TO_IMAGES = Path('images')
PATH_TO_MODELS = Path('models')
PATH_TO_GRAPHS = Path('dnn_files')

image_paths = [img for img in PATH_TO_IMAGES.glob('**/*') if img.is_file()]

path_to_graph = str(PATH_TO_MODELS / PATH_TO_GRAPHS / 'frozen_inference_graph.pb')
path_to_graph_def = str(PATH_TO_MODELS / PATH_TO_GRAPHS / 'inference_graph_config.pbtxt')

for image_path in image_paths:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = image.copy()

    # Read the graph and run with OpenCV:
    cvNet = cv2.dnn.readNetFromTensorflow(path_to_graph, path_to_graph_def)

    imH, imW, _ = img.shape
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

    # Run the model
    start_time = time.time()
    cvOut = cvNet.forward()
    end_time = time.time()
    print("Inference time: ", end_time-start_time)
