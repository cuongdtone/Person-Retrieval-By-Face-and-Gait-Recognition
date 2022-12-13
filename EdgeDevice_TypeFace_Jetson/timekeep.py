# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/14/2022

from utils.task import FaceCameras
import onnxruntime
import logging
import sys
logging.disable(sys.maxsize)


def main(args):
    onnxruntime.set_default_logger_severity(3)
    session = FaceCameras(args)
    session.run()


if __name__ == '__main__':
    main('settings.yaml')
