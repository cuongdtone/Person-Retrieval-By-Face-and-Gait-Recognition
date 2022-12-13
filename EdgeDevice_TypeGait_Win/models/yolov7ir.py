# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/12/2022

import cv2
import numpy as np
from openvino.inference_engine import IECore


class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


class YOLOv7IR:

    def __init__(self, model="src/ir_model/yolo/yolov7-tiny_640x640.xml", conf_thres=0.7, iou_thres=0.5):
        self.has_postprocess = False
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        # core = Core()
        #
        # model_ir = core.read_model(model=model)
        # self.compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")
        # self.output_layer_ir = self.compiled_model_ir.output(0)

        path_to_xml_file = 'src/ir_model/yolo/yolov7-tiny_640x640.xml'
        path_to_bin_file = 'src/ir_model/yolo/yolov7-tiny_640x640.bin'
        ie = IECore()
        net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)
        self.exec_net = ie.load_network(network=net, device_name="GPU.0", num_requests=1)

        self.input_width, self.input_height = (640, 640)

    def __call__(self, image):
        return self.detect_objects(image)

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        # Perform inference on the image
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.exec_net.infer({'images': input_tensor})
        return outputs['output']

    def process_output(self, output):
        predictions = np.squeeze(output[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)
        # return boxes, scores, class_ids
        return boxes[indices], scores[indices], class_ids[indices]

    def parse_processed_output(self, outputs):

        scores = np.squeeze(outputs[0], axis=1)
        predictions = outputs[1]
        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        if len(scores) == 0:
            return [], [], []

        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img

# # -*- coding: utf-8 -*-
# # @Organization  : TMT
# # @Author        : Cuong Tran
# # @Time          : 10/12/2022
#
# import time
# import cv2
# import numpy as np
# # import onnxruntime
# from openvino.runtime import Core
# import openvino
#
# class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#
# # Create a list of colors for each class where each color is a tuple of 3 integer values
# rng = np.random.default_rng(3)
# colors = rng.uniform(0, 255, size=(len(class_names), 3))
#
#
# class YOLOv7IR:
#
#     def __init__(self, model="src/ir_model/yolo/yolov7-tiny_640x640.xml", conf_thres=0.7, iou_thres=0.5):
#         self.has_postprocess = False
#         self.conf_threshold = conf_thres
#         self.iou_threshold = iou_thres
#
#         # Initialize model
#         core = Core()
#
#         model_ir = core.read_model(model=model)
#         self.compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")
#         self.output_layer_ir = self.compiled_model_ir.output(0)
#
#         self.input_width, self.input_height = (640, 640)
#
#     def __call__(self, image):
#         return self.detect_objects(image)
#
#     def detect_objects(self, image):
#         input_tensor = self.prepare_input(image)
#         # Perform inference on the image
#         outputs = self.inference(input_tensor)
#         self.boxes, self.scores, self.class_ids = self.process_output(outputs)
#         return self.boxes, self.scores, self.class_ids
#
#     def prepare_input(self, image):
#         self.img_height, self.img_width = image.shape[:2]
#
#         input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Resize input image
#         input_img = cv2.resize(input_img, (self.input_width, self.input_height))
#
#         # Scale input pixel values to 0 to 1
#         input_img = input_img / 255.0
#         input_img = input_img.transpose(2, 0, 1)
#         input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
#         return input_tensor
#
#     def inference(self, input_tensor):
#         outputs = self.compiled_model_ir([input_tensor])[self.output_layer_ir]
#         return outputs
#
#     def process_output(self, output):
#         predictions = np.squeeze(output[0])
#
#         # Filter out object confidence scores below threshold
#         obj_conf = predictions[:, 4]
#         predictions = predictions[obj_conf > self.conf_threshold]
#         obj_conf = obj_conf[obj_conf > self.conf_threshold]
#
#         # Multiply class confidence with bounding box confidence
#         predictions[:, 5:] *= obj_conf[:, np.newaxis]
#
#         # Get the scores
#         scores = np.max(predictions[:, 5:], axis=1)
#
#         # Filter out the objects with a low score
#         predictions = predictions[scores > self.conf_threshold]
#         scores = scores[scores > self.conf_threshold]
#
#         if len(scores) == 0:
#             return [], [], []
#
#         # Get the class with the highest confidence
#         class_ids = np.argmax(predictions[:, 5:], axis=1)
#
#         # Get bounding boxes for each object
#         boxes = self.extract_boxes(predictions)
#
#         # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
#         indices = nms(boxes, scores, self.iou_threshold)
#         # return boxes, scores, class_ids
#         return boxes[indices], scores[indices], class_ids[indices]
#
#     def parse_processed_output(self, outputs):
#
#         scores = np.squeeze(outputs[0], axis=1)
#         predictions = outputs[1]
#         # Filter out object scores below threshold
#         valid_scores = scores > self.conf_threshold
#         predictions = predictions[valid_scores, :]
#         scores = scores[valid_scores]
#
#         if len(scores) == 0:
#             return [], [], []
#
#         # Extract the boxes and class ids
#         # TODO: Separate based on batch number
#         batch_number = predictions[:, 0]
#         class_ids = predictions[:, 1]
#         boxes = predictions[:, 2:]
#
#         # In postprocess, the x,y are the y,x
#         boxes = boxes[:, [1, 0, 3, 2]]
#
#         # Rescale boxes to original image dimensions
#         boxes = self.rescale_boxes(boxes)
#
#         return boxes, scores, class_ids
#
#     def extract_boxes(self, predictions):
#         # Extract boxes from predictions
#         boxes = predictions[:, :4]
#
#         # Scale boxes to original image dimensions
#         boxes = self.rescale_boxes(boxes)
#
#         # Convert boxes to xyxy format
#         boxes = xywh2xyxy(boxes)
#
#         return boxes
#
#     def rescale_boxes(self, boxes):
#
#         # Rescale boxes to original image dimensions
#         input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
#         boxes = np.divide(boxes, input_shape, dtype=np.float32)
#         boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
#         return boxes
#
#     def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
#
#         return draw_detections(image, self.boxes, self.scores,
#                                self.class_ids, mask_alpha)
#
#     def get_input_details(self):
#         model_inputs = self.session.get_inputs()
#         self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
#
#         self.input_shape = model_inputs[0].shape
#         self.input_height = self.input_shape[2]
#         self.input_width = self.input_shape[3]
#
#     def get_output_details(self):
#         model_outputs = self.session.get_outputs()
#         self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
