FROM windj007/jupyter-keras-tools
MAINTAINER Roman <windj007@gmail.com>

RUN pip install tabula-py
RUN apt-get update && apt-get -f install -y openjdk-8-jdk scala parallel
RUN wget -O /tmp/sbt-0.13.15.deb https://dl.bintray.com/sbt/debian/sbt-0.13.15.deb && \
    dpkg -i /tmp/sbt-0.13.15.deb && \
    rm /tmp/sbt-0.13.15.deb
RUN apt-get install -yqq imagemagick latexmk \
    texlive-base texlive-base-bin texlive-bibtex-extra texlive-binaries \
    texlive-context texlive-extra-utils texlive-font-utils texlive-fonts-extra \
    texlive-fonts-extra-doc texlive-fonts-extra texlive-fonts-recommended texlive-fonts-recommended-doc \
    texlive-fonts-recommended texlive-formats-extra texlive-full texlive-games \
    texlive-generic-extra texlive-generic-recommended texlive-htmlxml texlive-humanities \
    texlive-humanities-doc texlive-humanities texlive-lang-african texlive-lang-all \
    texlive-lang-arabic texlive-lang-chinese texlive-lang-cjk texlive-lang-cyrillic \
    texlive-lang-czechslovak texlive-lang-english texlive-lang-european texlive-lang-french \
    texlive-lang-german texlive-lang-greek texlive-lang-indic texlive-lang-italian \
    texlive-lang-japanese texlive-lang-korean texlive-lang-other texlive-lang-polish \
    texlive-lang-portuguese texlive-lang-spanish texlive-latex-base texlive-latex-base-doc \
    texlive-latex-base texlive-latex-extra texlive-latex-extra-doc texlive-latex-extra \
    texlive-latex-recommended texlive-latex-recommended-doc texlive-latex-recommended \
    texlive-latex3 texlive-luatex texlive-math-extra texlive-metapost texlive-metapost-doc \
    texlive-metapost texlive-music texlive-omega texlive-pictures texlive-pictures-doc \
    texlive-plain-extra texlive-pstricks texlive-pstricks-doc texlive-pstricks texlive-publishers \
    texlive-publishers-doc texlive-publishers texlive-science texlive-science-doc texlive-science \
    texlive-xetex latexmk
RUN pip install svgwrite Pillow opencv-python tabula-py pdfquery \
    git+https://github.com/windj007/TexSoup timeout-decorator ngram \
    git+https://github.com/windj007/imgaug
RUN pip uninstall -y imgaug && pip install -U imgaug
