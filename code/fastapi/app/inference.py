import argparse
import cv2
import numpy as np
from pycocotools.coco import COCO
from inference_function import *


# input : model_img(path list(str) 1.jpg~end.jpg), exam_info(str) : ex 2012_9_a
# output : dict{q(str) : a(str)}


def make_inference_model(
    model_type,
    model_info,
    coco_json_path,
    img_folder_path,
    imgs_path=["1.jpg"],
    exam_info="",
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

    Returns:
        inference_model Class
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

    inference_model = Inference(
        img_folder_path,
        imgs_path,
        exam_info,
        coco,
        detector,
    )
    return inference_model


if __name__ == "__main__":
    inference_model = make_inference_model(
        model_type="mmdetection",
        model_info=[
            "/opt/ml/input/code/work_dirs/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco.py",
            "/opt/ml/input/code/work_dirs/faster_rcnn_r50_fpn_fp16_1x_coco/best_bbox_mAP_epoch_90.pth",
        ],
        coco_json_path="/opt/ml/input/data/annotations/train_v1-3.json",
        img_folder_path="/opt/ml/input/code/fastapi/app/tmp",
        imgs_path=[f"{i}.jpg" for i in range(1, 13)],
        exam_info="2021_f_a_0",
    )
    print(inference_model.make_user_solution(is_save=True, log_save=True))
