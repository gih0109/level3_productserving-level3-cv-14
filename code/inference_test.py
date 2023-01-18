import argparse
import cv2
import numpy as np
from pycocotools.coco import COCO
from inference_function import *


# input : model_img(path list(str) 1.jpg~end.jpg), exam_info(str) : ex 2012_9_a
# output : dict{q(str) : a(str)}


def make_qadict(
    model_type,
    model_info,
    coco_json_path,
    img_folder_path,
    imgs_path,
    exam_info,
    is_save=False,
):
    """
    Args:
        model_type (str): mmdeploy, mmdetection
        model_info (List(str)):
                model config
                mmeploy    : [model_path]
                mmdetection: [config_file,checkpoint_file]
        coco_json_path (str): annotation json path
        img_folder_path (str): img_folder_path
        imgs_path (List(str)): [1.jpg,2.jpg, ...]
        exam_info (str): 시험 정보 ex) 2023_f_n_0
        is_save (bool) : predicted 한 img의 저장 여부, default = False

    Returns:
        dict: key 문제번호, value : 예측한 체크박스의 번호
    """
    assert model_type in [
        "mmdeploy",
        "mmdetection",
    ], "모델 타입은 mmdeploy 또는 mmdetection으로 입력해주세요."

    coco = COCO(coco_json_path)
    if model_type == "mmdeploy":
        from mmdeploy_python import Detector

        detector = Detector(model_path=model_info[0], device_name="cpu", device_id=0)
        Inference = MMdeployInference

    if model_type == "mmdetection":
        from mmdet.apis import init_detector

        detector = init_detector(model_info[0], model_info[1], device="cuda:0")
        Inference = MMdetectionInference

    model_inference = Inference(
        img_folder_path,
        imgs_path,
        exam_info,
        coco,
        detector,
    )
    question_answer = model_inference.make_question_answer(is_save=is_save)
    return question_answer


if __name__ == "__main__":
    make_qadict(
        model_type="mmdetection",
        model_info=[
            "/opt/ml/input/code/work_dirs/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco.py",
            "/opt/ml/input/code/work_dirs/faster_rcnn_r50_fpn_fp16_1x_coco/epoch_24.pth",
        ],
        coco_json_path="/opt/ml/input/data/annotations/train_v1-3.json",
        img_folder_path="/opt/ml/input/code",
        imgs_path=["1.jpg", "2.jpg", "3.jpg", "4.jpg"],
        exam_info="2023_f_n_0",
        is_save=True,
    )
