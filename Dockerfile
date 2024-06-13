FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

COPY requirements_dev.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install pybind11
RUN pip install cvxpy
RUN pip install riskfolio-lib
RUN pip install -r requirements.txt

COPY lwhf lwhf
COPY setup.py setup.py
RUN mkdir raw_data
COPY Makefile Makefile

RUN pip install .
RUN pip install protobuf==3.20.*
# RUN make reset_local_files

CMD uvicorn lwhf.api.fast:app --host 0.0.0.0 --port $PORT
