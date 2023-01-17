import argparse

import cv2
import numpy as np
from mmdeploy_python import Detector
from pycocotools.coco import COCO
from inference_function import *


def parse_args():
    parser = argparse.ArgumentParser(description="show how to use sdk python api")

    parser.add_argument(
        "--model",
        dest="model_path",
        help="path of mmdeploy SDK model dumped by model converter",
    )
    parser.add_argument(
        "--device",
        dest="device_name",
        default="cpu",
        help="name of device, cuda or cpu",
    )
    args = parser.parse_args()
    return args


# input : model_img(path list(str) 1.jpg~end.jpg), exam_info(str) : ex 2012_9_a
# 이미지 정보 가져오기 ... 1
# 디텍션 수행 ... 2
# 1~2 비교하여 정답 만들기 (죄측하단 기준으로 confi 가장 높은애 )
# output : dict{q(str) : a(str)}


def main():
    args = parse_args()
    ###################### 수정 필요한 초기값 ############################
    coco = COCO("/opt/ml/input/data/annotations/train_v1-3.json")
    img_folder_path = "/opt/ml/input/data/images"
    imgs_path = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    exam_info = "2023_f_n_0"
    detector = Detector(
        model_path=args.model_path, device_name=args.device_name, device_id=0
    )
    ##################################################################
    model_inference = Inference(
        img_folder_path,
        imgs_path,
        exam_info,
        coco,
        detector,
    )
    question_answer = model_inference.make_question_answer()


if __name__ == "__main__":
    main()
