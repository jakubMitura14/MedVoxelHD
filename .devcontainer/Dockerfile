# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

################################################################################
# Prevent apt-get from prompting for keyboard choice
#  https://superuser.com/questions/1356914/how-to-install-xserver-xorg-in-unattended-mode
ENV DEBIAN_FRONTEND=noninteractive

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
    build-essential \
    wget\
    manpages-dev\
    g++\
    gcc\
    nodejs\
    libssl-dev\
    unzip\
    libidn11-dev\
    libglu1-mesa-dev \
    # libpangox-1.0-dev \
    # libpangoxft-1.0-0 \
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
    && rm -rf /var/lib/apt/lists/*


# ENV PATH="/opt/cmake/bin:${PATH}"
ENV NB_USER sliceruser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 && \
    apt-get install -y  python3.10-dev



RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    g++ \
    gfortran \
    git \
    vim \
    #   hdf5-tools \
    libmkl-full-dev \ 
    libboost-dev \
    libfftw3-dev \
    libnfft3-dev \
    libgfortran5 \
    libgmp-dev \
    #   libhdf5-dev \
    libopenmpi-dev \
    cython3 \
    openmpi-bin \
    openmpi-common \
    openmpi-doc \
    python3-dev \
    python3-mako \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-mpi4py \
    python3-pip \
    python3-sphinx \
    python3-skimage \ 
    python3-nbsphinx \
    python3-notebook \
    libpython3-dev \
    git \
    ca-certificates \
    openssl \
    rsync \
    libomp-dev \
    nano \
    less \
    # Jupyter lab packages
    nodejs \
    #   texlive \
    #   dvipng \
    #   texlive-latex-extra \
    #   texlive-fonts-recommended \
    vim net-tools curl \
    libgl1-mesa-glx \
    xserver-xorg-video-dummy \
    libxrender1 \
    libpulse0 \
    libpulse-mainloop-glib0  \
    libnss3  \
    libxcomposite1 \
    libxcursor1 \
    libfontconfig1 \
    libxrandr2 \
    libasound2 \
    libglu1 \
    x11vnc \
    awesome \
    jq \
    git \
    nautilus \
    jupyter-core \
    unzip \
    default-libmysqlclient-dev

#   apt-get autoremove --purge -y && \
#   apt-get autoclean -y && \
#   rm -rf /var/cache/apt/* /var/lib/apt/lists/*


# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp


RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py



RUN python3 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 
# RUN python3 -m pip install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2--index-url https://download.pytorch.org/whl/cu121


RUN python3 -m pip  --no-cache-dir install install --no-cache-dir \
    pandas==1.5.3 \
    future>=0.17.1 \
    grpcio \
    mock \
    portpicker \
    requests \
    h5py==3.10.0\
    #itk \
    #SimpleITK>=2.1.1.2 \
    # pandas \
    pydicom-seg \
    numpy==1.26.1\
    opencv-python \
    pydicom \
    more-itertools \
    SimpleITK==2.3.0 \
    highdicom==0.19.0 \
    # nnunet==1.7.0 \
    multiprocess \
    six \
    pymia==0.3.2 \
    einops seaborn \
    optuna \
    torchio \
    connected-components-3d \
    ninja==1.11.1.1\
    cython==0.29.28 \
    decorator \
    jupyter \
    ipywidgets \
    pytest \
    pandas \
    shapely \
    monty \
    mpi4py \
    pymatgen \
    ase \ 
    Jinja2==3.0.3\
    Mako \
    Pandoc \
    # Jupyter lab packages
    jupyterlab \
    scikit-image \
    itk==5.3.0 \
    vtk \
    monai[nibabel,itk,tqdm]==1.3.0 \
    nibabel \
    llvmlite \
    numba==0.55.1


RUN python3 -m pip  --no-cache-dir install install --no-cache-dir gspread
RUN mkdir ${HOME}/code
RUN git clone https://github.com/jakubMitura14/Hausdorff_morphological.git ${HOME}/code

COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT [ "/docker-entrypoint.sh" ]
# CMD ["sleep", "infinity" ,"sh", "-c", "./Slicer/bin/PythonSlicer -m jupyter notebook --port=$JUPYTERPORT --ip=0.0.0.0 --no-browser --NotebookApp.default_url=/lab/"]
# CMD ["sleep", "infinity"]
################################################################################
# Install Slicer application startup script

# COPY .slicerrc.py .

