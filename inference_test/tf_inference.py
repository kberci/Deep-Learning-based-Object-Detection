import time
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path


PATH_TO_IMAGES = Path('images')
PATH_TO_MODELS = Path('models')
PATH_TO_GRAPHS = Path('dnn_files')

image_paths = [img for img in PATH_TO_IMAGES.glob('**/*') if img.is_file()]

path_to_graph = str(PATH_TO_MODELS / PATH_TO_GRAPHS / 'frozen_inference_graph.pb')

# Read the graph and run with tensorflow:
with tf.io.gfile.GFile(path_to_graph, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.compat.v1.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Run the model
    output_names = {
        'output_boxes': 'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',
        'output_scores': 'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',
        'output_labels': 'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0'
    }

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = image.copy()

        # Read and preprocess an image.
        imH, imW, _ = img.shape
        inp = cv2.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        start_time = time.time()
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        end_time = time.time()
        print("Inference time: ", end_time - start_time)
