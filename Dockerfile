FROM alpine:3.6

# Here we use several hacks collected from https://github.com/gliderlabs/docker-alpine/issues/11:
# 1. install GLibc (which is not the cleanest solution at all)
# 2. hotfix /etc/nsswitch.conf, which is apperently required by glibc and is not used in Alpine Linux

# Install glibc
RUN apk add --update wget ca-certificates && \
    wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://raw.githubusercontent.com/sgerrand/alpine-pkg-glibc/master/sgerrand.rsa.pub && \
    wget https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.25-r0/glibc-2.25-r0.apk && \
    apk add glibc-2.25-r0.apk && \
    echo 'hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4' >> /etc/nsswitch.conf && \
    apk del wget ca-certificates

    # && \
    # rm "$ALPINE_GLIBC_PACKAGE" "$ALPINE_GLIBC_BIN_PACKAGE" /var/cache/apk/*
    # /usr/glibc/usr/bin/ldconfig "/lib" "/usr/glibc/usr/lib" && \


# Install python compiled for glibc
RUN apk add --update bash wget ca-certificates && \
    wget "http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm ./Miniconda3-latest-Linux-x86_64.sh && \
    ln -s ../miniconda/bin/* /usr/local/bin/ && \
    apk del bash wget && \
    rm /var/cache/apk/*

#RUN /usr/glibc/usr/bin/ldconfig /lib /usr/local/miniconda/lib
ENV PATH /usr/local/miniconda/bin:$PATH
RUN conda install -y python==3.6
RUN conda install -y numpy libgcc freetype pyzmq swig scipy
# jupyter matplotlib requests ipykernel
# RUN python -m ipykernel.kernelspec
RUN pip install https://pypi.python.org/packages/7d/d0/96269b9ecfcc55cb38779831595e0521c34ef4ecdeba08b1ba4194cc4813/tensorflow-1.2.1-cp36-cp36m-manylinux1_x86_64.whl#md5=525ab154b9e177f84fc7e15a38f891ff

RUN pip install deepcut

COPY keras.json /root/.keras/

WORKDIR /root

ENTRYPOINT /bin/sh
