import time
import glob
import tflite_runtime.interpreter as tflite
import tflite_utils
import os
import pandas as pd


PATH_TO_IMAGES = './images/'
PATH_TO_CONFIG = './config/'
PATH_TO_LOG = './results/inference_times.csv'
CONF_THRESHOLD = .4
is_coral_plugged = True
only_inference_test = True


# Define path to images and grab all paths
image_paths = [i.replace('\\', '/') for i in glob.glob(r'' + PATH_TO_IMAGES + '/*')]

# Initialize log file for inference times
test_index = tflite_utils.get_log_index(PATH_TO_LOG)

# Get model types and file names
folders = ['edgetpu', 'mobilenet_models', 'mobilenet', 'mobiledet', 'yolo4', 'yolo5', 'yolo5_models']

for folder in folders[:1]:
    print("\nFolder: " + folder)
    PATH_TO_MODELS = './models/' + folder + '/'
    models = [{'name': model, 'type': 'mobilenet'} for model in os.listdir(PATH_TO_MODELS)]
    # d = pd.read_csv(PATH_TO_CONFIG + folder + '.csv', squeeze=True)
    # models = [{d.columns[i]: item[i] for i in range(len(d.columns))} for item in d.values]
    for MODEL in models:
        # Overwrite name:
        # MODEL['name'] = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
        print('\n' + MODEL['name'] + ':')

        # Load the Tensorflow Lite model
        if is_coral_plugged:
          model_file, *device = MODEL['name'].split('@')
          delegate = [tflite.load_delegate(tflite_utils.EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})] \
              if 'edge' in MODEL['name'] else None
          interpreter = tflite.Interpreter(model_path=PATH_TO_MODELS + MODEL['name'], experimental_delegates=delegate)
        else:
          if 'edge' not in MODEL['name']:
            interpreter = tflite.Interpreter(model_path=PATH_TO_MODELS + MODEL['name'])
          else:
            print("Coral is not plugged, skipping " + MODEL['name'])
            continue
        interpreter.allocate_tensors()
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for image_path in image_paths:
            # Load image and resize to expected shape [1xHxWx3]
            image, input_data = tflite_utils.prepare_input_data(image_path, input_details, MODEL['type'])
            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run the model
            start_time = time.time()
            interpreter.invoke()
            inference_time = str(round(time.time() - start_time, 4))
            print("\tInference time: " + inference_time + " s")

            if only_inference_test:
                # Save results
                tflite_utils.save_only_inference_time(
                    test_index, MODEL, PATH_TO_LOG, image_path, inference_time)

            else:
                # Retrieve detection results
                boxes, classes, scores = tflite_utils.get_detection_results(
                    interpreter, input_details,
                    output_details, CONF_THRESHOLD, MODEL['type'])
                # Draw boxes
                image_result = tflite_utils.draw_boxes(
                    input_details, image, boxes, classes, scores, CONF_THRESHOLD,
                    MODEL['type'])
                # Save results
                tflite_utils.save_results(image_result, test_index, MODEL,
                                          PATH_TO_LOG, image_path, inference_time)

