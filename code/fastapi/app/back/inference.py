from mmdet.apis import inference_detector
from PIL import Image
from copy import deepcopy
from utils import *
from recognition import *
import cv2
import os


class Inference:
    def __init__(self, images, exam_info, coco, detector):
        self.images = images
        self.coco = coco
        self.detector = detector
        self.exam_info = self.load_exam_info(exam_info)
        self.inference_detector = inference_detector

    def load_exam_info(self, exam_info):
        """
        img info를 불러오는 함수

        Args:
            exam_info (str): 시험지의 정보 {년도}_{몇월}_{형}
                            ex) 2013_9_a or 2022_f

            coco : 사전에 제작된 annotation기반 coco format 데이터

        Returns:
            List: exam_info에 대응하는 img파일들의 사전 정보
            ex) : exam_info = 2012_9_a 일때 12년 9월 a형에 문제 정보가 p1 ~ 마지막 페이지까지 리스트에 담김
                [2012_9_a_p1 정보, 2012_9_a_p2 정보 .....]
        """
        all_img_info = self.coco.loadImgs(self.coco.getImgIds())
        result = [info for info in all_img_info if exam_info in info["file_name"]]
        result = sorted(result, key=lambda x: x["file_name"])
        return result

    def load_anns(self, idx):
        img_info = self.exam_info[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_info["id"])
        anns = self.coco.loadAnns(ann_ids)
        return img_info, anns

    def load_anns_q(self, idx):
        """
        시험의 정보와 이미지를 받아 그것에 대응되는 사전에 annotation된 questions을 반환
        """
        _, anns = self.load_anns(idx)
        questions = [ann for ann in anns if ann["category_id"] > 6]
        anns_dict = {q["category_id"]: self.xywh2ltrb(q["bbox"]) for q in questions}
        return anns_dict

    def get_predict(self, img, box_threshold=0.1):
        """_summary_

        Args:
            img: img 파일의 경로
            detector: 사전에 제작된 mmdetection 모델
            box_threshold (float, optional): detector로 예측된 box들의 최소 임계값

        Returns:
            List: [np.array(left,top,right,bottom,label) .... ]
            list안의 값은 np.array 형식으로 박스정보와 label 값이 int type으로 정의됨
        """
        inference = self.inference_detector(self.detector, img)
        predict = []
        for label, bboxes in enumerate(inference):
            predict += [
                np.append(bbox[:4], [bbox[4], label])
                for bbox in bboxes
                if (bbox[4] > box_threshold)
            ]
        predict = sorted(predict, key=lambda x: x[4], reverse=True)
        return predict

    def xywh2ltrb(self, bbox):
        """
        Args: (x,y)는 좌측 상단
            bbox: (x,y,w,h)

        Returns:
            bbox: (left,top,right,bottom)
        """
        return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

    def compute_iou(self, box1, box2):
        """iou 계산

        Args:
            box :(List): [left,top,right,bottom]

        Returns:
            iou(float): box1과 box2의 iou값 반환
        """
        x1 = np.maximum(box1[0], box2[0])
        y1 = np.maximum(box1[1], box2[1])
        x2 = np.minimum(box1[2], box2[2])
        y2 = np.minimum(box1[3], box2[3])

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box1_area + box2_area - intersection

        iou = intersection / union
        assert iou <= 1, "iou값이 1 초과"
        return iou

    def match_qa(self, anns):
        """
        사전에 정의된 annotation을 기반으로 같은 문제의 question 박스와 answer 박스를 매칭
        Args:
            anns : 사전에 정의된 annotation 정보

        Returns:
            anns_dict (dict): 문제에 대응하는 정답 박스의 위치 반환
            key : category_id
            value : answer box의 정보 : left,top,right,bottom
        """

        questions = [ann for ann in anns if ann["category_id"] > 6]
        answers = [ann for ann in anns if ann["category_id"] <= 6]

        # 추후에 중복되는 연산 제거하는 코드로 수정하기!!
        anns_dict = {}
        for q in questions:
            q_ = self.xywh2ltrb(q["bbox"])
            for a in answers:
                a_ = self.xywh2ltrb(a["bbox"])
                if self.compute_iou(q_, a_) > 0:
                    anns_dict[q["category_id"]] = a_
                    break

        return anns_dict

    def make_qa(self, predict, anns):
        """문제에 대응하는 최적의 answer box 찾기
           최적의 answer box 기준은 사전의 annotation결과와 iou가 가장 높은 box
           iou 값이 0.3 이상인 박스가 없는경우 예측을 못했다고 판단하고 결과로 (0,0,0,0,0,0)을 반환

        Args:
            predict (List): [[box_info, confidence score,label] ....] box_info = left,top,right,bottom
            anns (dict): key : 문제 categori id, value: bbox left,top,right,bottom

        Returns:
            question_answer (dict):  key : 문제번호, value : [boxinfo, categori_id]
        """
        question_answer = {}
        for q, bbox in anns.items():
            iou_list = [self.compute_iou(pred, bbox) for pred in predict]
            max_iou = max(iou_list) if iou_list else 0
            if max_iou < 0.3:
                question_answer[q - 6] = np.array([0, 0, 0, 0, 0, -1])
            else:
                question_answer[q - 6] = predict[iou_list.index(max_iou)]
        return question_answer

    def resize_box(self, q_a_box, img_info, img_shape):
        """input의 이미지 size와 사전에 정의된 annotation의 box사이즈를 대응시킴

        Args:
            q_a_box : dict : key = 문제, value : 문제에 대응하는 정답박스 정보
            img_info : 사전에 정의된 annotation 이미지 정보
            img_shape : input 이미지의 사이즈

        Returns:
            annotation의 size를 적용한
            q_a_box : dict : key = 문제, value : 문제에 대응하는 정답박스 정보
        """
        w_r = img_shape[1] / img_info["width"]
        h_r = img_shape[0] / img_info["height"]

        q_a_box = {
            q: (int(a[0] * w_r), int(a[1] * h_r), int(a[2] * w_r), int(a[3] * h_r))
            for q, a in q_a_box.items()
        }
        return q_a_box

    def save_predict(self, img, img_path, qa_info):
        """
        predicted img 저장하는 함수!
        """
        for bbox in qa_info.values():
            left, top, right, bottom = bbox[:4].astype(int)
            cv2.putText(
                img,
                f"{int(bbox[-1])}   {bbox[-2]:.4f}",
                (left, top - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (255, 0, 0),
                3,
            )
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
            cv2.imwrite(
                f"/opt/ml/input/code/fastapi/app/log/{img_path}_predict.jpg", img
            )

    def make_user_solution(self, img_save=False, log_save=False):
        """
        Args:
            img_save (bool): 예측한 결과의 사진을 저장할지 여부
            log_save(bool): 이미지의 정답과 예측값 그 값에 대응하는 confidence csv 저장

        Returns:
            dict: key 문제번호, value : 예측한 체크박스의 번호
        """
        answer_bbox, a_label = {}, []
        images = deepcopy(self.images)
        for idx, img in enumerate(images):
            img_info, anns = self.load_anns(idx)
            predict = self.get_predict(img)
            q_a_box = self.match_qa(anns)
            q_a_box = self.resize_box(q_a_box, img_info, img.shape)
            qa_info = self.make_qa(predict, q_a_box)
            answer_bbox.update(qa_info)

            if img_save:
                self.save_predict(img, idx, qa_info)

            if log_save:
                a_label += [ann for ann in anns if ann["category_id"] <= 6]

        if log_save:
            a_b = sorted(answer_bbox.items())
            pred_data = pd.DataFrame(
                {
                    "label": [a["category_id"] - 1 for a in a_label],
                    "predict": [int(bbox[-1]) for q, bbox in a_b],
                    "confidence": [bbox[-2] for q, bbox in a_b],
                }
            )
            if not os.path.isdir("/opt/ml/input/code/fastapi/app/log"):
                os.mkdir("/opt/ml/input/code/fastapi/app/log")
            pred_data.to_csv("/opt/ml/input/code/fastapi/app/log/predict_log.csv")

        user_solution = {q: int(bbox[-1]) for q, bbox in sorted(answer_bbox.items())}
        return user_solution


def save_score_img(self, scoring_result):
    # 채점된 이미지를 만들기 위해 o, x 이미지를 불러오는 부분입니다.
    # TODO: 위의 input이미지의 resize 부분과 함께 고려해야 할 사항입니다.
    o_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/correct.png")
    x_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/wrong.png")
    o_width, o_height = o_image.size
    x_width, x_height = x_image.size

    # TODO: 현재 paste 좌표가 좌측 하단으로 잡혀있음 (좌측 상단으로 바꿔야함. annotation 정보 확인 필요)
    score_img = []
    for idx, img in enumerate(self.images):  # fix
        background = Image.fromarray(img)
        question_ann = self.load_anns_q(idx)
        for cat_id, bbox in question_ann.items():
            question = cat_id - 6  # 문제 번호: 1 ~ 30
            if scoring_result[question] == "O":
                background.paste(
                    o_image,
                    (
                        int(bbox[0] - o_width / 2),
                        int(bbox[1] - o_height / 2) + 10,
                    ),
                    o_image,
                )
            elif scoring_result[question] == "X":
                background.paste(
                    x_image,
                    (
                        int(bbox[0] - x_width / 2),
                        int(bbox[1] - x_height / 2) + 10,
                    ),
                    x_image,
                )
        score_img.append(background)
    return score_img


# if __name__ == "__main__":
#     from pycocotools.coco import COCO
#     from mmdet.apis import init_detector
#     import os
#     from PIL import Image

#     model_config = "/opt/ml/input/data/models/19/config.py"
#     model_weight = "/opt/ml/input/data/models/19/model.pth"
#     coco = COCO("/opt/ml/input/data/annotations/train_v1-3.json")
#     detector = init_detector(model_config, model_weight, device="cuda:0")

#     img_path = "/opt/ml/input/data/infer_test"
#     img_paths = [os.path.join(img_path, img) for img in os.listdir(img_path)]
#     images_np = [np.array(Image.open(img)) for img in img_paths]
#     inference = Inference(
#         images=images_np,
#         exam_info="2021_f_a",
#         coco=coco,
#         detector=detector,
#     )
#     result = inference.make_user_solution(True, True)
#     print("hi")


class Inference_v2:
    def __init__(self, images, detector, q_bbox, answer, img_shape, time, ocr_model):
        self.images = images
        self.detector = detector
        self.q_bbox = q_bbox
        self.answer = answer
        self.inference_detector = inference_detector
        self.origin_img_shape = img_shape
        self.time = time
        self.ocr_model = ocr_model

    def ltrb2xywh(self, bbox):
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

    def get_predict(self, img, box_threshold=0.1):
        inference = self.inference_detector(self.detector, img)
        predict = []
        for label, bboxes in enumerate(inference):
            for bbox in bboxes:
                if bbox[-1] > box_threshold:
                    predict.append(list(bbox) + [label])
        return predict

    # 사각형 겹침 여부 확인
    def overlap(self, bbox, patch):
        l1, t1, r1, b1 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        l2, t2, r2, b2 = patch[0], patch[1], patch[0] + patch[2], patch[1] + patch[3]

        overlap = not (r1 <= l2 or l1 >= r2 or b1 <= t2 or t1 >= b2)
        return overlap  # True면 겹침

    def resize_box(self, bbox, img_shape):
        w_r = img_shape[1] / self.origin_img_shape[0]
        h_r = img_shape[0] / self.origin_img_shape[1]

        bbox = [
            int(bbox[0] * w_r),
            int(bbox[1] * h_r),
            int(bbox[2] * w_r),
            int(bbox[3] * h_r),
        ]
        return bbox

    def save_predict(self, img, img_path, bbox):
        x, y, w, h = map(int, bbox[:4])
        confidence = bbox[4]
        pred = bbox[5]
        cv2.putText(
            img,
            f"{pred}   {confidence:.4f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.9,
            (255, 0, 0),
            3,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.imwrite(
            f"/opt/ml/input/code/fastapi/app/log/{self.time}/{img_path}_predict.jpg",
            img,
        )

    def save_score_img(self, scoring_result):
        # 채점된 이미지를 만들기 위해 o, x 이미지를 불러오는 부분입니다.
        # TODO: 위의 input이미지의 resize 부분과 함께 고려해야 할 사항입니다.
        o_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/correct.png")
        x_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/wrong.png")
        o_width, o_height = o_image.size
        x_width, x_height = x_image.size

        # TODO: 현재 paste 좌표가 좌측 하단으로 잡혀있음 (좌측 상단으로 바꿔야함. annotation 정보 확인 필요)
        score_img = []
        for idx, img in enumerate(self.images):  # fix
            background = Image.fromarray(img)
            question = self.q_bbox[idx][0]
            img_shape = img.shape
            for q in question:
                bbox = self.resize_box(question[q], img_shape)
                if scoring_result[q] == "O":
                    background.paste(
                        o_image,
                        (
                            int(bbox[0] - o_width / 2),
                            int(bbox[1] - o_height / 2) + 10,
                        ),
                        o_image,
                    )
                elif scoring_result[q] == "X":
                    background.paste(
                        x_image,
                        (
                            int(bbox[0] - x_width / 2),
                            int(bbox[1] - x_height / 2) + 10,
                        ),
                        x_image,
                    )
            score_img.append(background)
        return score_img

    def is_in(self, box, x_min, y_min):  # 좌측 하단 점이 box 안에 포함되는지?
        return box[0] < x_min < box[0] + box[2] and box[1] < y_min < box[1] + box[3]

    def main(self):
        images = deepcopy(self.images)
        result = dict()
        log_pred = []
        for idx, img in enumerate(images):
            predict = self.get_predict(img)

            for i in range(len(predict)):
                predict[i] = self.ltrb2xywh(predict[i][:4]) + predict[i][4:]
            question = self.q_bbox[idx][0]
            resize_question = {}
            for q, v in question.items():
                resize_q = self.resize_box(question[q], img.shape)
                resize_question[q] = resize_q
                tmp = []
                for p in predict:
                    if self.overlap(resize_q, p[:4]):
                        p.append(q)
                        tmp.append(p)
                if tmp:
                    tmp = sorted(tmp, key=lambda x: x[4])
                    pred = tmp[-1]
                    result[q] = pred[-2]
                    log_pred.append(pred)
                    self.save_predict(img, idx, pred)
                else:
                    result[q] = -1

            # 시험지의 좌측, 우측 을 나누는 코드
            h, w, c = img.shape
            middle_line = w / 2
            left_questions = []
            right_questions = []
            for q in resize_question:
                if resize_question[q][0] < middle_line:  # top-left의 x좌표가 중앙선보다 작으면 왼쪽에
                    left_questions.append(resize_question[q])
                else:
                    right_questions.append(resize_question[q])
            # y좌표 기준으로 정렬
            left_questions.sort(key=lambda x: x[1])
            right_questions.sort(key=lambda x: x[2])

            areas = []  # 문제 구역 나눔
            for i, box in enumerate(left_questions):
                if i == (len(left_questions) - 1):
                    areas.append(
                        [0, box[1] + box[3], middle_line, h - (box[1] + box[3])]
                    )
                else:
                    areas.append(
                        [
                            0,
                            box[1] + box[3],
                            middle_line,
                            left_questions[i + 1][1] - (box[1] + box[3]),
                        ]
                    )
            for i, box in enumerate(right_questions):
                if i == (len(right_questions) - 1):
                    areas.append(
                        [
                            middle_line,
                            box[1] + box[3],
                            middle_line,
                            h - (box[1] + box[3]),
                        ]
                    )
                else:
                    areas.append(
                        [
                            middle_line,
                            box[1] + box[3],
                            middle_line,
                            right_questions[i + 1][1] - (box[1] + box[3]),
                        ]
                    )

            for i, q in enumerate(resize_question):
                if result[q] == 6 or result[q] == -1:
                    candidates = [
                        pred
                        for pred in predict
                        if self.is_in(areas[i], pred[0], pred[1] + pred[3])
                    ]  # 미리 정해진 구역에 있는 예측값 모두 저장
                    if len(candidates) > 1:
                        candidates.sort(key=lambda x: x[4], reverse=True)
                    if candidates:
                        x, y, w, h = list(map(int, candidates[0]))[:4]
                        cv2.imwrite(
                            f"/opt/ml/input/code/fastapi/app/tmp/{q}.jpg",
                            img[y - 10 : (y + h) + 10, x - 10 : (x + w) + 10, :],
                        )
                        ocr_result = text_recognition(
                            model=self.ocr_model,
                            img_folder="/opt/ml/input/code/fastapi/app/tmp",
                            device="cuda:0",
                        )
                        result[q] = int(
                            ocr_result[f"/opt/ml/input/code/fastapi/app/tmp/{q}.jpg"]
                        )
                        # os.system("rm /opt/ml/input/code/fastapi/app/tmp/*")
        print(result)
        tmp = [i for i in range(1, 31)]
        for x in result:
            if x in tmp:
                tmp[x - 1] = 0
        for x in tmp:
            if x != 0:
                result[x] = -1

        _score = score(result, self.answer)
        scoring_img = self.save_score_img(_score)
        return scoring_img, log_pred
