#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

FROM nvidia/cuda:11.3.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    # build-essential \
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
 && rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y wget

RUN mkdir /app
WORKDIR /app

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda-11.3



#- nvidia::cudatoolkit=11.3.1

# Create a working directory


#RUN apt-get install linux-headers-$(uname -r)
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN apt-get install -y cuda-toolkit-11-3


# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/home/user/miniconda/bin:$PATH
COPY environment.yml /app/environment.yml
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda env update -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && conda clean -ya
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN pip install pymia
RUN pip install gspread
RUN pip install nibabel
RUN pip install itk
RUN pip install monai
RUN pip install gspread
# RUN pip install warp-lang==0.3.1
RUN pip install pytorch-lightning
RUN pip install numba
# RUN pip install cupy


RUN git clone https://github.com/jakubMitura14/HausMorphoCode.git /home/data/hausDorffCode
#RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
# USER root
#ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"
# USER user
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
# RUN git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew \
# && mkdir ~/.linuxbrew/bin \
# && ln -s ../Homebrew/bin/brew ~/.linuxbrew/bin \
# && eval $(~/.linuxbrew/bin/brew shellenv) \
# && brew --version
# RUN export PATH=$HOME/.linuxbrew/bin:$PATH


# directory = /home/data/spleenDat

# warpLoss/spleenWarp.py

# from monai.config import print_config
# from monai.apps import download_and_extract
# import torch
# import matplotlib.pyplot as plt
# import tempfile
# import shutil
# import os
# import glob

# root_dir =’ /home/data/spleenDat’
# resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
# md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

# compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
# data_dir = os.path.join(root_dir, "Task09_Spleen")
# if not os.path.exists(data_dir):
#     download_and_extract(resource, compressed_file, root_dir, md5)