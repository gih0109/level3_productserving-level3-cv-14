import os

import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from clova_ocr.dataset import RawDataset, AlignCollate
from clova_ocr.utils import AttnLabelConverter
from clova_ocr.model import Model


"""
주의:
    이 코드는 TPS-ResNet-BiLSTM-Attn 모델 용으로 작성된 코드입니다.
"""

# load text recognition model funtion
def load_ocr_model(
    img_scale: tuple = (32, 100),  # (height, width)
    num_fiducial: int = 20,
    input_channel: int = 1,
    output_channel: int = 512,
    hidden_size: int = 256,
    character: str = "0123456789abcdefghijklmnopqrstuvwxyz",
    batch_max_length: int = 25,
    Transformation: str = "TPS",
    FeatureExtraction: str = "ResNet",
    SequenceModeling: str = "BiLSTM",
    Prediction: str = "Attn",
    save_model: str = None,
    device: str = "cpu",
):
    """
    text recognition model 을 불러오는 함수

    Args:
        ### 입력 필요 Args ###
        save_model(str) : 모델 weight 가 저장된 pth 파일 경로
        device(str) : cuda or cpu
    """

    assert save_model is not None
    assert Transformation == "TPS" and FeatureExtraction == "ResNet"
    assert SequenceModeling == "BiLSTM" and Prediction == "Attn"

    converter = AttnLabelConverter(character)
    num_class = len(converter.character)

    print(f"loading pretrained model from {save_model}")

    model = Model(
        Transformation,
        FeatureExtraction,
        SequenceModeling,
        Prediction,
        num_fiducial,
        img_scale,
        input_channel,
        output_channel,
        hidden_size,
        num_class,
        batch_max_length,
    )
    model = torch.nn.DataParallel(model).to(device)
    # load model
    model.load_state_dict(torch.load(save_model, map_location=device))
    print("model load sucessifully")

    return model


def text_recognition(
    model=None,
    img_folder: str = None,
    img_scale: tuple = (32, 100),  # (height, width)
    rgb: bool = False,
    batch_size: int = 192,
    batch_max_length: int = 25,
    workers: int = 4,
    keep_ratio_with_pad: bool = False,
    character: str = "0123456789abcdefghijklmnopqrstuvwxyz",
    device: str = "cpu",
) -> dict:
    """
    데이터셋 생성 & 모델 인퍼런스 함수

    Args:
        ### 입력 필요 Args ###
        model: load_ocr_model 에 의해 로드된 모델(인스턴스)
        img_folder(str): 이미지가 저장된 폴더 경로
        device(str): cpu or cuda

    return:
        dictionary
        {각 이미지 경로: 인식된 텍스트}
    """

    assert model is not None
    assert img_folder is not None

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(
        imgH=img_scale[0], imgW=img_scale[1], keep_ratio_with_pad=keep_ratio_with_pad
    )
    ocr_data = RawDataset(
        root=img_folder, img_scale=img_scale, rgb=rgb
    )  # use RawDataset
    ocr_data_loader = torch.utils.data.DataLoader(
        ocr_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(workers),
        collate_fn=AlignCollate_demo,
        pin_memory=True,
    )

    converter = AttnLabelConverter(character)

    # predict
    model.eval()
    with torch.no_grad():
        result = {}
        for image_tensors, image_path_list in ocr_data_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(
                device
            )
            text_for_pred = (
                torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)
            )

            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(
                image_path_list, preds_str, preds_max_prob
            ):
                # if 'Attn' in Prediction:
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                # confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                result[img_name] = pred

        return result
