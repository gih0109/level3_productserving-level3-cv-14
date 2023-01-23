import numpy as np
import pandas as pd
import cv2
import os


class MMdeployInference:
    def __init__(self, img_folder_path, imgs_path, exam_info, coco, detector):
        self.img_folder_path = img_folder_path
        self.imgs_path = imgs_path
        self.coco = coco
        self.detector = detector
        self.exam_info = self.load_exam_info(exam_info, coco)

    def update_exam_info(self, exam_info):
        self.exam_info = self.load_exam_info(exam_info, self.coco)

    def load_anns(self, img_path):
        img_info = self.exam_info[int(img_path.split(".")[0]) - 1]
        ann_ids = self.coco.getAnnIds(imgIds=img_info["id"])
        anns = self.coco.loadAnns(ann_ids)
        return img_info, anns

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

    def load_anns_q(self, exam_info, img_path, coco):
        """
        시험의 정보와 이미지를 받아 그것에 대응되는 사전에 annotation된 questions을 반환
        Args:
            exam_info (str): 체첨하려는 시험의 정보
            img_path (str): 쪽수가 적혀있는 img의 경로, ex) 3.jpg : 3쪽의 이미지
            coco : 사전에 제작된 annotation기반 coco format 데이터
            annotation box 정보 : x,y,w,h
        Returns:
            anns_dict (dict): 문제에 대응하는 정답 박스의 위치 반환
            key : category_id
            value : question box의 정보 : left,top,right,bottom
        """
        img_info = exam_info[int(img_path.split(".")[0]) - 1]
        ann_ids = coco.getAnnIds(imgIds=img_info["id"])
        anns = coco.loadAnns(ann_ids)
        questions = [ann for ann in anns if ann["category_id"] > 6]
        answers = [ann for ann in anns if ann["category_id"] <= 6]

        # 추후에 중복되는 연산 제거하는 코드로 수정하기!!
        anns_dict = {}
        for q in questions:
            q_ = self.xywh2ltrb(q["bbox"])
            for a in answers:
                a_ = self.xywh2ltrb(a["bbox"])
                if self.compute_iou(q_, a_) > 0:
                    anns_dict[q["category_id"]] = q_
                    break

        return anns_dict

    def load_exam_info(self, exam_info, coco):
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
        all_img_info = coco.loadImgs(coco.getImgIds())
        result = [info for info in all_img_info if exam_info in info["file_name"]]
        result = sorted(result, key=lambda x: x["file_name"])
        return result

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

    def get_predict(self, img, detector, box_threshold=0.3):
        """_summary_

        Args:
            img: cv2.imread로 읽은 이미지 데이터
            detector: 사전에 제작된 detector 모델
            box_threshold (float, optional): detector로 예측된 box들의 최소 임계값

        Returns:
            List: [np.array(left,top,right,bottom,class_id) .... ]
            list안의 값은 np.array 형식으로 박스정보와 label 값이 int type으로 정의됨
        """
        bboxes, labels, _ = detector(img)
        predict = [
            np.append(bbox, label)
            for bbox, label in zip(bboxes, labels)
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
                question_answer[q - 6] = np.array([0, 0, 0, 0, 0, 0])
            else:
                question_answer[q - 6] = predict[iou_list.index(max_iou)]
        return question_answer

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
            cv2.imwrite(f"{img_path}_predict.jpg", img)

    def make_question_answer(self, is_save=False):
        answer_bbox = {}
        for img_path in self.imgs_path:
            img = cv2.imread(os.path.join(self.img_folder_path, img_path))
            predict = self.get_predict(img, self.detector)
            anns = self.load_anns(self.exam_info, img_path, self.coco)
            qa_info = self.make_qa(predict, anns)
            answer_bbox.update(qa_info)
            if is_save:
                self.save_predict(img, img_path, qa_info)
        question_answer = {q: bbox[-1] for q, bbox in sorted(answer_bbox.items())}
        return question_answer


class MMdetectionInference(MMdeployInference):
    def __init__(self, img_folder_path, imgs_path, exam_info, coco, detector):
        from mmdet.apis import inference_detector

        super().__init__(img_folder_path, imgs_path, exam_info, coco, detector)
        self.inference_detector = inference_detector

    def get_predict(self, img, detector, box_threshold=0.3):
        """_summary_

        Args:
            img: img 파일의 경로
            detector: 사전에 제작된 mmdetection 모델
            box_threshold (float, optional): detector로 예측된 box들의 최소 임계값

        Returns:
            List: [np.array(left,top,right,bottom,label) .... ]
            list안의 값은 np.array 형식으로 박스정보와 label 값이 int type으로 정의됨
        """
        inferece = self.inference_detector(detector, img)
        predict = []
        for label, bboxes in enumerate(inferece):
            predict += [
                np.append(bbox[:4], [bbox[4], label])
                for bbox in bboxes
                if (bbox[4] > box_threshold)
            ]
        predict = sorted(predict, key=lambda x: x[4], reverse=True)
        return predict

    def make_user_solution(self, is_save=False, log_save=False):
        """
        Args:
            is_save (bool): 예측한 결과의 사진을 저장할지 여부
            log_save(bool): 이미지의 정답과 예측값 그 값에 대응하는 confidence csv 저장

        Returns:
            dict: key 문제번호, value : 예측한 체크박스의 번호
        """
        answer_bbox, a_label = {}, []
        for img_path in self.imgs_path:
            img_info, anns = self.load_anns(img_path)
            img = cv2.imread(os.path.join(self.img_folder_path, img_path))

            predict = self.get_predict(img, self.detector)
            q_a_box = self.match_qa(anns)
            q_a_box = self.resize_box(q_a_box, img_info, img.shape)
            qa_info = self.make_qa(predict, q_a_box)
            answer_bbox.update(qa_info)

            if is_save:
                self.save_predict(img, img_path, qa_info)

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
            pred_data.to_csv("./predict_log.csv")

        user_solution = {q: int(bbox[-1]) for q, bbox in sorted(answer_bbox.items())}
        return user_solution
