FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

# Install 3rd party apps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/
# Define a build-time argument for image type
ARG IMAGE_TYPE=full

RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY . /workspace
RUN python initialize.py

EXPOSE 8000