FROM ubuntu:18.04

WORKDIR /app
COPY . .


RUN  apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get -y --no-install-recommends install \
    sudo \
    vim \
    wget \
    build-essential \
    pkg-config \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python-dev \
    python3-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0
RUN apt-get -y install python3-pip && pip3 install --upgrade pip

RUN apt-get -y --no-install-recommends install \
    git \
    cmake \
    autoconf \
    automake \
    libtool \
    gstreamer-1.0 \
    gstreamer1.0-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    python-gst-1.0 \
    libgirepository1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libcairo2-dev \
    gir1.2-gstreamer-1.0 \
    python3-gi \
    python-gi-dev \
    python3-setuptools

RUN pip3 install numpy setuptools==59.6.0 setuptools-scm wheel Cython netifaces python-dotenv
RUN python3.8 -m pip install --ignore-installed PyGObject
RUN apt-get install build-essential autoconf libtool pkg-config python-opengl python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev -y
RUN pip3 install greenlet gevent
RUN apt install libjpeg-dev zlib1g-dev -y
RUN apt install build-essential libdbus-glib-1-dev libgirepository1.0-dev
RUN apt-get install redis libicu-dev python3-distutils-extra libudev-dev libsystemd-dev libxml2-dev libcups2-dev libxmlsec1-dev -y
RUN pip3 install -r requirements.txt
RUN python3 -m pip install detectron2 -f \https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update -y
RUN apt-get install -y kmod
RUN apt-get install wget && wget https://dist.ipfs.tech/kubo/v0.17.0/kubo_v0.17.0_linux-amd64.tar.gz
RUN tar -xvzf kubo_v0.17.0_linux-amd64.tar.gz && cd kubo && ./install.sh
RUN cd /app
#ARG nvidia_binary_version="525.60.11"
#ARG nvidia_binary="NVIDIA-Linux-x86_64-${nvidia_binary_version}.run"
#RUN wget -q https://us.download.nvidia.com/XFree86/Linux-x86_64/${nvidia_binary_version}/${nvidia_binary} && chmod +x ${nvidia_binary} && ./${nvidia_binary} --accept-license --ui=none --no-kernel-module --no-questions && rm -rf ${nvidia_binary}

CMD ["python3", "Pipeline.py"]
