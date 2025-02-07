FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    zip \
    unzip \
    libgl1-mesa-glx \
    git \
    xauth \
    x11-apps \
    python3-pyqt5

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN pip3 install flask flask-cors gunicorn

RUN git clone https://github.com/yatengLG/ISAT_with_segment_anything.git && \
    cd ISAT_with_segment_anything && \
    pip install -r requirements.txt

ENV DISPLAY=:0
WORKDIR /app
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["bash", "/app/entrypoint.sh"]



