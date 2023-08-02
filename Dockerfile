#FROM python:3.8
FROM  nvidia/cuda:11.7.1-runtime-ubuntu20.04
WORKDIR /sbert_es
COPY requirements.txt requirements.txt
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone
RUN apt-get update -y && apt-get install python3 python3-pip curl libgl1 libglib2.0-0 -y  && apt-get clean
RUN pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/


EXPOSE 8000