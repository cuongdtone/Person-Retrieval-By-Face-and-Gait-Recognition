# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/12/2022

import numpy as np
import cv2
from unidecode import unidecode


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def normalize(v):
    v = v - v.min()
    return v, v.max()


def plot(x, y):
    x = np.array(x)
    y = np.array(y)
    x, x_max = normalize(x)
    y, y_max = normalize(y)
    size = max(x_max, y_max) * 12
    if size == 0:
        size = 100
    image = np.zeros((size, size, 3), dtype='uint8')

    x = 10+x*9.5
    y = 10+y*9.5
    lines = np.array([x, y]).T.astype('int')
    cv2.polylines(image, [lines], isClosed=False, thickness=2, color=(255, 0, 0))
    # for x_, y_ in zip(x, y):
    #     image[10+int(x_*9.5), 10+int(y_*9.5)] = 255
    return image


def plot_yolo(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


def plot_tracking(image, people):
    im = np.ascontiguousarray(np.copy(image))

    text_scale = 1
    text_thickness = 2
    line_thickness = 3
    name_show = ''
    for obj_id, person in people.items():
        body_box = person['box']
        name = person['name']
        if name_show == '' and name is not None:
            name_show = name
        text = f'{obj_id}-{unidecode(name.split()[-1] if name is not None else "Unknown")}'

        (label_width, label_height), baseline = cv2.getTextSize(text,
                                                                cv2.FONT_HERSHEY_PLAIN,
                                                                text_scale, text_thickness)
        color = get_color(abs(obj_id))

        x1_b, y1_b, x2_b, y2_b = body_box
        cv2.rectangle(im, (x1_b, y1_b), (x1_b + label_width, y1_b + label_height), color, -1)
        cv2.rectangle(im, (x1_b, y1_b), (x2_b, y2_b), color, 2)

        cv2.putText(im, text, (x1_b, y1_b + label_height), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255),
                    thickness=text_thickness)

    return im, name_show
