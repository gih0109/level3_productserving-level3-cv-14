import numpy as np
import cv2
import os


class Inference:
    def __init__(self, img_folder_path, imgs_path, exam_info, coco, detector):
        self.img_folder_path = img_folder_path
        self.imgs_path = imgs_path
        self.coco = coco
        self.detector = detector
        self.exam_info = self.load_exam_info(exam_info, coco)

    def load_anns(self, exam_info, img_path, coco):
        """
        시험의 정보와 이미지를 받아 그것에 대응되는 사전에 annotation된 anns을 반환

        Args:
            exam_info (str): 체첨하려는 시험의 정보
            img_path (str): 쪽수가 적혀있는 img의 경로, ex) 3.jpg : 3쪽의 이미지
            coco : 사전에 제작된 annotation기반 coco format 데이터

        Returns:
            anns_dict (dict): 문제에 대응하는 정답 박스의 위치 반환
            key : category_id
            value : answer box의 정보
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
                if self.compute_iou(q_, self.xywh2ltrb(a["bbox"])) > 0:
                    anns_dict[q["category_id"]] = a["bbox"]
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
        # Calculate intersection areas
        x1 = np.maximum(box1[0], box2[0])
        y1 = np.maximum(box1[3], box2[3])
        x2 = np.minimum(box1[2], box2[2])
        y2 = np.minimum(box1[1], box2[1])

        box1_area = (box1[2] - box1[0]) * (box1[1] - box1[3])
        box2_area = (box2[2] - box2[0]) * (box2[1] - box2[3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box1_area + box2_area - intersection

        iou = intersection / union
        return iou

    def get_predict(self, img, detector, box_threshold=0.3):
        """_summary_

        Args:
            img: cv2.imread로 읽은 이미지 데이터
            detector: 사전에 제작된 detector 모델
            box_threshold (float, optional): detector로 예측된 box들의 최소 임계값

        Returns:
            List: [np.array(left,top,right,bottom,class_id) .... ]
            list안의 값은 np.array 형식으로 x,y,w,h의 박스정보와 classid 값이 int type으로 정의됨
        """
        bboxes, labels, _ = detector(img)
        predict = [
            np.append(bbox[:4], label).astype(int)
            for bbox, label in zip(bboxes, labels)
            if (bbox[4] > box_threshold and label > 0)
        ]
        return predict

    def xywh2ltrb(self, bbox):
        """
        Args:
            bbox: (x,y,w,h)

        Returns:
            bbox: (left,top,right,bottom)
        """
        return bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1]

    def ltrb2xywh(self, bbox):
        """
        Args:
            bbox: (left,top,right,bottom)

        Returns:
            bbox: (x,y,w,h)
        """
        return bbox[0], bbox[3], bbox[2] - bbox[0], bbox[1] - bbox[3]

    def make_qa(self, predict, anns):
        question_answer = {}
        for q, bbox in anns.items():
            question_answer[q - 6] = max(
                predict,
                key=lambda x: self.compute_iou(
                    self.xywh2ltrb(x[:4]), self.xywh2ltrb(bbox)
                ),
            )
        return question_answer

    def save_predict(self, img, qa_info):
        for q, bbox in qa_info.items():
            left, top, right, bottom = self.xywh2ltrb(bbox[:4])
            cv2.putText(
                img,
                str(bbox[-1]),
                (left, top - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (0, 255, 0),
                3,
            )
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.imwrite("output_detection.png", img)

    def make_question_answer(self, is_save=False):
        answer_bbox = {}
        for img_path in self.imgs_path:
            img = cv2.imread(os.path.join(self.img_folder_path, img_path))
            predict = self.get_predict(img, self.detector)
            anns = self.load_anns(self.exam_info, img_path, self.coco)
            qa_info = self.make_qa(predict, anns)
            answer_bbox.update(qa_info)
            if is_save:
                self.save_predict(img, qa_info)
        question_answer = {q: bbox[-1] for q, bbox in sorted(answer_bbox.items())}
        return question_answer
