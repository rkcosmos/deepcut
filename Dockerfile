FROM tensorflow/tensorflow:2.0.0-py3

RUN  apt-get install -y git \
     && pip install --upgrade pip \
     && pip install git+https://github.com/rkcosmos/deepcut.git

ENTRYPOINT /bin/bash
