FROM ubuntu:14.04
MAINTAINER Mehran Pesteie <mehranp@ece.ubc.ca>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -f -b -p $CONDA_DIR && \
    rm ~/anaconda.sh

# Needed for OpenCV
RUN apt-get install -y libgtk2.0-0 git

RUN useradd -m deepinfer && echo "deepinfer:deepinfer" | chpasswd && adduser deepinfer sudo

RUN mkdir -p $CONDA_DIR && \
    chown deepinfer $CONDA_DIR -R

USER deepinfer

RUN pip install --upgrade pip && \
    conda install -c menpo opencv=2.4.11 && \
    conda clean -yt

RUN mkdir -p /home/deepinfer/git && \
    cd /home/deepinfer/git && \
    git clone https://github.com/mpslxz/SpineClassifierDocker && \
    cd SpineClassifierDocker && \
    cd Python && \
    mkdir -p data

WORKDIR "/home/deepinfer"
ENTRYPOINT ["python", "/home/deepinfer/git/SpineClassifierDocker/Python/numpy_model.py"]