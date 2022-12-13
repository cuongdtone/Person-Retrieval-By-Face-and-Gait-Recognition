import numpy as np
import onnxruntime
import cv2
from openvino.runtime import Core


class FaceMask:
    def __init__(self):
        model_file = 'src/ir_model/face_mask/face_mask_cls.xml'
        core = Core()
        model_ir = core.read_model(model=model_file)
        self.compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")

        self.output_layer_ir = self.compiled_model_ir.output(0)

    def __call__(self, image_face):
        """
        iamge_face: norm crop face
        Kiểm tra khuôn mặt có đeo khẩu trang hay không
        :param image_face: hình ảnh ( np.array ) face
        :return: True / False ( Không đeo / Đeo )
        """
        h, w = image_face.shape[:2]
        image_face = image_face[h // 6:h, w // 5:w - w // 5]
        image_face = cv2.resize(cv2.cvtColor(image_face, cv2.COLOR_BGR2GRAY), (16, 16))
        image_face = np.expand_dims(np.expand_dims(image_face, axis=2), axis=0).astype(np.float32)
        pred = self.compiled_model_ir([image_face])
        pred = pred[self.output_layer_ir]
        return np.argmax(pred) == 1
