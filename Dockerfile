FROM python:3.10.6-buster
FROM tensorflow/tensorflow:2.16.1

COPY requirements_dev.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY lwhf /lwhf
COPY setup.py /setup.py
COPY Makefile Makefile

RUN pip install .
RUN pip install protobuf==3.20.*
# RUN make reset_local_files

CMD uvicorn lwhf.api.fast:app --host 0.0.0.0 --port $PORT
