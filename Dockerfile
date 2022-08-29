FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update
RUN apt-get install build-essential git -y

RUN git clone https://github.com/MICLab-Unicamp/medseg

RUN pip install medseg/.

RUN medseg --help
