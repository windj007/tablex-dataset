#!/bin/bash

mkdir -p /notebook/data/1_pdffigures2_out /notebook/2_page_images

cd /tmp/pdffigures2
sbt "run-main org.allenai.pdffigures2.FigureExtractorBatchCli /notebook/data/0_source_pdfs/ -e -m /notebook/data/1_pdffigures2_out/ -d /notebook/data/1_pdffigures2_out/"
cd /notebook

for fname in /notebook/data/0_source_pdfs/*.pdf
do
    fname=${fname%%.pdf}
    fname=${fname##*/}
    convert /notebook/data/0_source_pdfs/$fname.pdf /notebook/data/2_page_images/$fname-%04d.png
done

./make_training_data.py
./make_inout_pairs.py
./train_test_augment.py