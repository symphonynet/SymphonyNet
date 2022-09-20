FROM nvidia/cuda:11.7.1-base-ubuntu20.04

CMD nvidia-smi

# Declare some ARGuments
ARG PYTHON_VERSION=3.6
ARG CONDA_VERSION=3
# Installation of some libraries / RUN some commands on the base image
ARG CONDA_PY_VERSION=4.5.11

# Base system
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip python3-dev wget binutils g++ bash \
    bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# CONDA
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda$CONDA_VERSION-$CONDA_PY_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# Create a conda environment to use the h2o4gpu
RUN /usr/bin/chmod +x /usr/bin/tini

# You can add the new created environment to the path
RUN conda update -n base -c defaults conda 
# && conda create -y -n gpuenvs -c h2oai -c conda-forge h2o4gpu-cuda9


# Copy the files in the actual directory from the directory forDocker on our host into the container in the directory /testenv
#ENV PATH /opt/conda/envs/gpuenvs/bin:$PATH
# Set the working directory to be /testenv
COPY . /SymphonyNet

RUN . ~/.bashrc && \
    conda create -n SymphonyNet python=3.8 && \
    conda activate SymphonyNet

# cd path_to_your_env
# git clone this_project
# cd SymphonyNet
ENV FORCE_CUDA "1"
RUN pip install torch
RUN pip install pytorch-fast-transformers
RUN cd SymphonyNet && \
    pip install -r requirements.txt 
#    cat requirements.txt | xargs -n 1 -L 1 pip install

WORKDIR /SymphonyNet
