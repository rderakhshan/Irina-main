FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TEXMFVAR=/tmp/irina-texmf/var
ENV TEXMFCONFIG=/tmp/irina-texmf/config
ENV TEXMFHOME=/tmp/irina-texmf/home

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-fonts-extra \
        texlive-science \
        lmodern \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "pypdf>=5"

