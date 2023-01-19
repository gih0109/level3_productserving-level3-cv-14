from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from app.model import MyEfficientNet, get_model, get_config, predict_from_image_byte

import pandas as pd

app = FastAPI()

# TODO: 전체적으로 변성윤님 코드를 그대로 가져와서 사용중입니다.
# 불필요한 부분들은 걷어내고, 필요한 부분만 응용하여 우리의 상황에 맞게 바꾸는 작업이 필요할 것 같습니다.
# 정답지를 받아오기 위해, 모든 페이지가 하드코딩 되어 있습니다. -> 상황에 맞게 원하는 정답지를 찾아올 수 있도록 변환이 필요합니다.

orders = []


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    price: float


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])

    def add_product(self, product: Product):
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self


class OrderUpdate(BaseModel):
    products: List[Product] = Field(default_factory=list)


class InferenceImageProduct(Product):
    name: str = "inference_image_product"
    price: float = 100.0
    result: Optional[List]


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders


@app.get("/order/{order_id}", description="Order 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    order = get_order_by_id(order_id=order_id)
    if not order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return order


def get_order_by_id(order_id: UUID) -> Optional[Order]:
    return next((order for order in orders if order.id == order_id), None)


@app.post("/order", description="주문을 요청합니다")
async def make_order(
    files: List[UploadFile] = File(...),
    model: MyEfficientNet = Depends(get_model),
    config: Dict[str, Any] = Depends(get_config),
):
    products = []
    for file in files:
        image_bytes = await file.read()
        inference_result = predict_from_image_byte(model=model, image_bytes=image_bytes, config=config)
        product = InferenceImageProduct(result=inference_result)
        products.append(product)

    new_order = Order(products=products)
    orders.append(new_order)
    return new_order


def update_order_by_id(order_id: UUID, order_update: OrderUpdate) -> Optional[Order]:
    """
    Order를 업데이트 합니다

    Args:
        order_id (UUID): order id
        order_update (OrderUpdate): Order Update DTO

    Returns:
        Optional[Order]: 업데이트 된 Order 또는 None
    """
    existing_order = get_order_by_id(order_id=order_id)
    if not existing_order:
        return

    updated_order = existing_order.copy()
    for next_product in order_update.products:
        updated_order = existing_order.add_product(next_product)

    return updated_order


@app.patch("/order/{order_id}", description="주문을 수정합니다")
async def update_order(order_id: UUID, order_update: OrderUpdate):
    updated_order = update_order_by_id(order_id=order_id, order_update=order_update)

    if not updated_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return updated_order


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    found_order = get_order_by_id(order_id=order_id)
    if not found_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return found_order.bill


# TODO: 정답지를 받아오기 위해, 모든 페이지가 하드코딩 되어 있습니다. -> 상황에 맞게 원하는 정답지를 찾아올 수 있도록 변환이 필요합니다.
# TODO: 정답지 csv 파일 자체도 추가가 필요합니다. 현재 2021 수능 가형 정답지만이 확보되어 있습니다.
@app.get("/2021_f_a")
def read_info():
    answer_df = pd.read_csv("/opt/ml/input/code/fastapi/app/answer/2021_f_a.csv")
    question = answer_df["문항번호"].tolist()
    answer = answer_df["정답"].tolist()
    return {
        "answer": {f"{question[i]}": f"{answer[i]}" for i in range(len(question))},
        "points": answer_df["배점"].tolist(),
    }


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    read_info = pd.read_csv(files)
    question = list(read_info["question"])
    answer = read_info["answer"]
    return {
        "filenames": [file.filename for file in files],
        "question": question,
        "answer": answer,
    }


@app.get("/2021-06/11")
def read_info2106():
    """
    '2021-06-가형(홀수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-06/12")
def read_info2106():
    """
    '2021-06-가형(짝수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-06/21")
def read_info2106():
    """
    '2021-06-나형(홀수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-06/22")
def read_info2106():
    """
    '2021-06-나형(짝수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-09/11")
def read_info2106():
    """
    '2021-09-가형(홀수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-09/12")
def read_info2106():
    """
    '2021-09-가형(짝수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-09/21")
def read_info2106():
    """
    '2021-09-나형(홀수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-09/22")
def read_info2106():
    """
    '2021-09-나형(짝수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-f/11")
def read_info2106():
    """
    '2021-수능-가형(홀수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = {
        1: "3",
        2: "2",
        3: "1",
        4: "4",
        5: "5",
        6: "2",
        7: "1",
        8: "2",
        9: "4",
        10: "2",
        11: "1",
        12: "4",
        13: "3",
        14: "3",
        15: "2",
        16: "5",
        17: "3",
        18: "3",
        19: "5",
        20: "5",
        21: "2",
        22: "15",
        23: "8",
        24: "60",
        25: "160",
        26: "36",
        27: "13",
        28: "72",
        29: "201",
        30: "29",
    }
    return {"question": question, "answer": answer}


@app.get("/2021-f/12")
def read_info2106():
    """
    '2021-수능-가형(짝수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-f/21")
def read_info2106():
    """
    '2021-수능-나형(홀수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}


@app.get("/2021-f/22")
def read_info2106():
    """
    '2021-수능-나형(짝수)'
    """
    question = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "q23",
        "q24",
        "q25",
        "q26",
        "q27",
        "q28",
        "q29",
        "q30",
    ]
    answer = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a0",
        "a1",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a3",
        "a5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a0",
        "a1",
        "a2",
        "a4",
        "a5",
    ]
    return {"question": question, "answer": answer}
