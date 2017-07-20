#!/bin/bash

mkdir -p /notebook/data/0_source_pdfs /notebook/data/1_pdffigures2_out /notebook/data/2_page_images /notebook/data/3_prepared_images /notebook/data/4_inout_pairs

# python /notebook/pdffigures2/evaluation/download_from_urls.py -g -c
# cp /notebook/pdffigures2/evaluation/datasets/conference/pdfs/* /notebook/data/0_source_pdfs
# cp /notebook/pdffigures2/evaluation/datasets/s2/pdfs/* /notebook/data/0_source_pdfs


# cd /notebook/pdffigures2
# sbt "run-main org.allenai.pdffigures2.FigureExtractorBatchCli /notebook/data/0_source_pdfs/ -e -m /notebook/data/1_pdffigures2_out/ -d /notebook/data/1_pdffigures2_out/"
# cd /notebook

pdfs=`
for fname in $(ls /notebook/data/0_source_pdfs/*.pdf)
do
    fname=${fname%%.pdf}
    fname=${fname##*/}
    echo $fname
done
`

# parallel -j20 convert -define pdf:use-cropbox=true /notebook/data/0_source_pdfs/{}.pdf /notebook/data/2_page_images/{}-%04d.png ::: $pdfs

# ./convert_gold_pdffigures2_markup.py /notebook/data/0_source_pdfs /notebook/data/1_pdffigures2_out /notebook/pdffigures2/evaluation/datasets/conference/annotations.json
# ./convert_gold_pdffigures2_markup.py /notebook/data/0_source_pdfs /notebook/data/1_pdffigures2_out /notebook/pdffigures2/evaluation/datasets/s2/annotations.json

# ./make_training_data.py

cp /notebook/data/relabeled/* /notebook/data/3_prepared_images

./make_inout_pairs.py
./train_test_augment.py
