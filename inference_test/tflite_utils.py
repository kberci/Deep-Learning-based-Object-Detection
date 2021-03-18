import numpy as np
import cv2
from datetime import datetime
import platform
import os
import pandas as pd
import random
import colorsys
try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow is not installed.")
    print("TFLite inference time tests assumed without visualizing outputs.")


EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def prepare_input_data(image_path, input_details, model_type):
    # Get quantization and image details
    floating_model = input_details[0]['dtype'] == np.float32
    _, height, width, _ = list(input_details[0]['shape'])

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model and model_type != 'yolo4':
        input_mean, input_std = 127.5, 127.5
        input_data = (np.float32(input_data) - input_mean) / input_std
    elif model_type == 'yolo4':
        input_data = np.float32(input_data) / 255

    return image, input_data


def get_log_index(log_path):
    if not os.path.isfile(log_path):
        df = pd.DataFrame(columns=['test_index', 'system', 'date', 'start_time', 'model_name', 'model_type',
                                   'input_image_name', 'output_image_name', 'inference_time'])
        df.to_csv(log_path, sep=',', index=False, header=True)
    df = pd.read_csv(log_path)
    index = df.values[-1][0] + 1 if df.values.tolist() else 0
    return index


def save_only_inference_time(index, model, log_path, image_path, inference_time):
    filename = model['name'].split('.tflite')[0] + '_result-' + image_path.split('/')[-1].split('.')[0] + '.' + \
               image_path.split('/')[-1].split('.')[-1]  # Output filename
    # Save log info
    index = str(index)
    systems = {'Linux-4.19.112+-x86_64-with-Ubuntu-18.04-bionic': 'colab',
               'Windows-10-10.0.19041-SP0': 'laptop',
               'Linux-5.4.51-v7l+-armv7l-with-debian-10.4': 'raspberry',
               'Linux-5.4.0-53-generic-x86_64-with-debian-buster-sid': 'home_laptop'
               }
    try:
        system = systems[platform.platform()]
    except:
        system = "unkown"
    date, time = datetime.now().strftime("%d/%m/%Y"), datetime.now().strftime("%H:%M:%S")
    model_name, model_type = model['name'], model['type']
    input_image = image_path.split('/')[-1]
    output_image = filename
    df = pd.DataFrame([[index, system, date, time, model_name, model_type, input_image, output_image, inference_time]])
    df.to_csv(log_path, mode='a', header=False, index=False)


# def filter_boxes(box_xywh, scores, score_threshold=0.25, input_shape=tf.constant([416, 416])):
#     scores_max = tf.math.reduce_max(scores, axis=-1)
#     mask = scores_max >= score_threshold
#     class_boxes = tf.boolean_mask(box_xywh, mask)
#     pred_conf = tf.boolean_mask(scores, mask)
#     class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
#     pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])
#
#     box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)
#
#     input_shape = tf.cast(input_shape, dtype=tf.float32)
#
#     box_yx = box_xy[..., ::-1]
#     box_hw = box_wh[..., ::-1]
#
#     box_mins = (box_yx - (box_hw / 2.)) / input_shape
#     box_maxes = (box_yx + (box_hw / 2.)) / input_shape
#     boxes = tf.concat([
#         box_mins[..., 0:1],  # y_min
#         box_mins[..., 1:2],  # x_min
#         box_maxes[..., 0:1],  # y_max
#         box_maxes[..., 1:2]  # x_max
#     ], axis=-1)
#     return boxes, pred_conf


def get_object_name(class_id, model):
    with open('config/coco_labels.txt', 'r') as f:
        if model == 'yolo4':
            return ' '.join(f.readlines()[int(class_id)].split()[1:]), int(class_id)
        if model == 'yolo5':
            return ' '.join(f.readlines()[int(class_id)].split()[1:]), int(class_id)
        elif model == 'mobilenet':
            class_id += 1
        for i, line in enumerate(f):
            key, val = line.split()[0], ' '.join(line.split()[1:])
            if class_id == int(key):
                return val, i


def get_colors(num_classes):
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    return colors


# def get_detection_results(interpreter, input_details, output_details, confidence_threshold, model_type):
#     _, height, width, _ = list(input_details[0]['shape'])
#
#     if model_type == 'efficientdet':
#         boxes = interpreter.get_tensor(output_details[0]['index'])[0][:, 1:5]  # Bounding box coordinates
#         classes = interpreter.get_tensor(output_details[0]['index'])[0][:, 6]  # Class index
#         scores = interpreter.get_tensor(output_details[0]['index'])[0][:, 5]  # Confidence
#     elif model_type == 'mobilenet':
#         boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
#         classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
#         scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
#     elif model_type == 'yolo4':
#         pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
#         boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
#                                         input_shape=tf.constant([height, width]))
#         boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#             boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#             scores=tf.reshape(
#                 pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#             max_output_size_per_class=50,
#             max_total_size=50,
#             iou_threshold=0.5,
#             score_threshold=confidence_threshold
#         )
#         boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
#     elif model_type == 'yolo5':
#         # output_data = interpreter.get_tensor(output_details[0]['index'])
#         # pred = torch.tensor(output_data)
#         # pred = non_max_suppression(pred, 0.4, 0.5)
#         # boxes = list(pred[0][:, 0:4].numpy().astype(int)) if pred[0] is not None else []
#         # scores = list(pred[0][:, 4].numpy().astype(float)) if pred[0] is not None else []
#         # classes = list(pred[0][:, 5].numpy().astype(int)) if pred[0] is not None else []
#         boxes, scores, classes = [], [], []
#     else:
#         raise TypeError
#     return boxes, classes, scores


def draw_boxes(input_details, input_image, boxes, classes, scores, confidence_threshold, model_type):
    # Get sizes before and after resizing
    _, height, width, _ = list(input_details[0]['shape'])
    imH, imW, _ = input_image.shape

    # Get colors for visualization
    colors = get_colors(num_classes=80)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > confidence_threshold) and (scores[i] <= 1.0):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within
            ymin_, xmin_, ymax_, xmax_ = boxes[i][0] * imH, boxes[i][1] * imW, boxes[i][2] * imH, boxes[i][3] * imW
            if model_type == 'efficientdet':
                ymin_ /= height
                xmin_ /= width
                ymax_ /= height
                xmax_ /= width
            ymin, xmin, ymax, xmax = int(max(1, ymin_)), int(max(1, xmin_)), int(min(imH, ymax_)), int(min(imW, xmax_))

            # Look up object name from "labels" array using class index
            object_name, color_index = get_object_name(classes[i], model=model_type)

            cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), colors[color_index], 2)

            # Draw label
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            # Text box
            cv2.rectangle(input_image, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), colors[color_index], cv2.FILLED)
            # Label text
            cv2.putText(input_image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return input_image


def save_results(image, index, model, log_path, image_path, inference_time):
    # Save image result
    filename = model['name'].split('.tflite')[0] + '_result-' + image_path.split('/')[-1].split('.')[0] + '.' + \
               image_path.split('/')[-1].split('.')[-1]  # Output filename
    if not os.path.isdir('./results/images'):
        os.mkdir('./results/images')
    cv2.imwrite('./results/images/' + filename, image)  # Saving the image

    # Save log info
    index = str(index)
    systems = {'Linux-4.19.112+-x86_64-with-Ubuntu-18.04-bionic': 'colab',
               'Windows-10-10.0.19041-SP0': 'laptop',
               'Linux-5.4.51-v7l+-armv7l-with-debian-10.4': 'raspberry'
               }
    try:
        system = systems[platform.platform()]
    except:
        system = 'vision_PC'
    date, time = datetime.now().strftime("%d/%m/%Y"), datetime.now().strftime("%H:%M:%S")
    model_name, model_type = model['name'], model['type']
    input_image = image_path.split('/')[-1]
    output_image = filename
    df = pd.DataFrame([[index, system, date, time, model_name, model_type, input_image, output_image, inference_time]])
    df.to_csv(log_path, mode='a', header=False, index=False)


# Yolo5 Torch functions
# def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
#     """Performs Non-Maximum Suppression (NMS) on inference results
#
#     Returns:
#          detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
#     """
#
#     nc = prediction[0].shape[1] - 5  # number of classes
#     xc = prediction[..., 4] > conf_thres  # candidates
#
#     # Settings
#     min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
#     max_det = 300  # maximum number of detections per image
#     time_limit = 10.0  # seconds to quit after
#     redundant = True  # require redundant detections
#     multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
#
#     t = time.time()
#     output = [None] * prediction.shape[0]
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence
#
#         # If none remain process next image
#         if not x.shape[0]:
#             continue
#
#         # Compute conf
#         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
#
#         # Box (center x, center y, width, height) to (x1, y1, x2, y2)
#         box = xywh2xyxy(x[:, :4])
#
#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
#             x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
#         else:  # best class only
#             conf, j = x[:, 5:].max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
#
#         # Filter by class
#         if classes:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
#
#         # If none remain process next image
#         n = x.shape[0]  # number of boxes
#         if not n:
#             continue
#
#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#         i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]
#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#                 iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#                 weights = iou * scores[None]  # box weights
#                 x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#                 if redundant:
#                     i = i[iou.sum(1) > 1]  # require redundancy
#             except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
#                 print(x, i, x.shape, i.shape)
#                 pass
#
#         output[xi] = x[i]
#         if (time.time() - t) > time_limit:
#             break  # time limit exceeded
#
#     return output
#
#
# def xywh2xyxy(x):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y
#
#
# def box_iou(box1, box2):
#     # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
#     """
#     Return intersection-over-union (Jaccard index) of boxes.
#     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#     Arguments:
#         box1 (Tensor[N, 4])
#         box2 (Tensor[M, 4])
#     Returns:
#         iou (Tensor[N, M]): the NxM matrix containing the pairwise
#             IoU values for every element in boxes1 and boxes2
#     """
#
#     def box_area(box):
#         # box = 4xn
#         return (box[2] - box[0]) * (box[3] - box[1])
#
#     area1 = box_area(box1.T)
#     area2 = box_area(box2.T)
#
#     # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#     inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
#     return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)