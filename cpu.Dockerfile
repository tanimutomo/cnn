ARG UBUNTU_VERSION=16.04
FROM ubuntu:${UBUNTU_VERSION}

WORKDIR /app
ADD . /app

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} install --upgrade \
    pip \
    setuptools

RUN ${PIP} install --trusted-host pypi.python.org -r cpu_requirements.txt
