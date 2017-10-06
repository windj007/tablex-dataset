#!/usr/bin/env python

from latex_dataset import *


def pdf2samples_apply(arc):
    try:
        pdf2samples(arc, './data/arxiv/inout_pairs/', get_table_info, aggregate_object_bboxes)
    except Exception as ex:
        print(arc, ex)


with open('./good_papers.lst', 'r') as f:
    good_papers = set(line.strip() for line in f)


joblib.Parallel(n_jobs=10)(
    joblib.delayed(pdf2samples_apply)(arc)
    for arc in good_papers
)
