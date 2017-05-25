FROM windj007/jupyter-keras-tools
MAINTAINER Roman <windj007@gmail.com>

RUN pip install tabula-py
RUN apt-get update && apt-get -f install -y openjdk-8-jdk scala
RUN wget -O /tmp/sbt-0.13.15.deb https://dl.bintray.com/sbt/debian/sbt-0.13.15.deb && \
    dpkg -i /tmp/sbt-0.13.15.deb && \
    rm /tmp/sbt-0.13.15.deb
RUN git clone https://github.com/allenai/pdffigures2 /tmp/pdffigures2 && \
    cd /tmp/pdffigures2 && \
    sbt
RUN apt-get install -yqq imagemagick
RUN pip install svgwrite Pillow
