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
USER user

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

RUN pip install pymia
RUN pip install gspread
RUN pip install nibabel
RUN pip install itk
RUN pip install monai
RUN pip install gspread
RUN pip install warp-lang==0.3.1

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
USER root
ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"
USER user

# RUN git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew \
# && mkdir ~/.linuxbrew/bin \
# && ln -s ../Homebrew/bin/brew ~/.linuxbrew/bin \
# && eval $(~/.linuxbrew/bin/brew shellenv) \
# && brew --version
# RUN export PATH=$HOME/.linuxbrew/bin:$PATH


RUN brew install enzyme

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates gnupg2 less sudo \
 # Spicy build and test dependencies.
 && apt-get install -y --no-install-recommends git cmake ninja-build ccache bison flex g++ libfl-dev zlib1g-dev libssl-dev jq locales-all make \
 # Spicy doc dependencies.
 && apt-get install -y --no-install-recommends python3 python3-pip python3-sphinx python3-sphinx-rtd-theme python3-setuptools python3-wheel doxygen \
 && pip3 install "btest>=0.66" pre-commit \
 # Install a recent CMake.
 && mkdir -p /opt/cmake \
 && curl -L https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0-Linux-x86_64.tar.gz | tar xzvf - -C /opt/cmake --strip-components 1 \
 && apt-get clean \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

#sudo apt install clang-11 libclang-11-dev llvm-11-tools llvm-11-dev
# RUN wget -O - https://apt.llvm.org/llvm.sh
# RUN pip install lit
# RUN git clone https://github.com/vgvassilev/clad.git ${HOME}/clad
# RUN mkdir ${HOME}/build_dir inst; cd ${HOME}/build_dir
# RUN cmake ${HOME}/clad -DClang_DIR=/usr/lib/llvm-11 -DLLVM_DIR=/usr/lib/llvm-11 -DCMAKE_INSTALL_PREFIX=../inst -DLLVM_EXTERNAL_LIT="``which lit``"
# RUN make && make install
RUN conda install -c conda-forge clad
RUN pip install pip install pytorch-lightning

USER user

#export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}$ 
#export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# RUN apt-get update
# RUN apt-get install build-essential


# Set the default command to python3
CMD ["python3"]
