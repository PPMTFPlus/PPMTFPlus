FROM centos:centos7.5.1804

LABEL maintainer="murakami.takao　takao-murakami at aist.go.jp"

RUN yum update -y && \
    yum install -y vim less git && \
    yum install -y wget make && \
    yum install -y epel-release


RUN yum groupinstall "Development Tools" -y

RUN yum install -y zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel tk-devel libffi-devel openssl-devel


# install c++ 11.2
RUN wget http://ftp.tsukuba.wide.ad.jp/software/gcc/releases/gcc-11.2.0/gcc-11.2.0.tar.gz && \
    tar xvzf gcc-11.2.0.tar.gz && \
    cd gcc-11.2.0 && \
    ./contrib/download_prerequisites && \
    ./configure --enable-languages=c,c++ --prefix=/usr/local/lib/gcc-11.2.0 --disable-bootstrap --disable-multilib && \
    make && \
    make install
 

# clone PPMTF repository
RUN cd /opt/ && \ 
    git clone https://github.com/PPMTFPlus/PPMTFPlus

# install cpp libraries


## install Stats 3.1.1
RUN wget https://github.com/kthohr/stats/archive/v3.1.1.tar.gz && \
    tar -xvf ./v3.1.1.tar.gz && \
    cp -r ./stats-3.1.1/include/* /opt/PPMTFPlus/cpp/include/

## install Eigen 3.3.7 library
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz && \
    tar -zxvf eigen-3.3.7.tar.gz && \
    cp -r ./eigen-3.3.7/Eigen /opt/PPMTFPlus/cpp/include/


## install Gcem 1.13.1 library
RUN wget https://github.com/kthohr/gcem/archive/v1.13.1.tar.gz && \
    tar -xvf ./v1.13.1.tar.gz && \
    cp -r ./gcem-1.13.1/include/* /opt/PPMTFPlus/cpp/include/


# install python 3.10.2
RUN wget https://www.python.org/ftp/python/3.10.4/Python-3.10.4.tar.xz && \
    tar xJf Python-3.10.4.tar.xz && \
    cd Python-3.10.4 && \
    yum install -y epel-release && \
    yum install -y openssl11 openssl11-devel && \
    export CFLAGS=$(pkg-config --cflags openssl11) && \
    export LDFLAGS=$(pkg-config --libs openssl11)  && \
    ./configure && \
    make && \ 
    make altinstall

# install python libraries
RUN pip3.10 install scipy==1.9.0 numpy==1.23.2 POT==0.8.2

# compile cpp source file
RUN cd /opt/PPMTFPlus/cpp && \
    make

# cleanup
RUN rm *.tar.gz && rm *.tar.xz && \
    echo "PATH=/usr/local/lib/gcc-11.2.0/bin:\$PATH" >> ~/.bash_profile && \
    echo "export PATH" >> ~/.bash_profile 
