FROM ubuntu:24.04
USER root

RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
      g++-14 \
      gfortran-14 \
      libgsl-dev \
      liblapacke-dev \
      pandoc \
      pkg-config \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      sudo \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14
ENV CXX=g++-14
ENV FC=gfortran-14

ENV PATH=/home/few/.local/few-venv/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
