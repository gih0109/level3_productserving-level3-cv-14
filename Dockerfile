FROM python:3.8

COPY ./code/fastapi code/fastapi

WORKDIR code/fastapi

RUN pip install --upgrade pip
RUN pip install -r front_requirements.txt
