#!/usr/bin/env python

import json, os, sys, numpy


BB_KEYS = 'x1 y1 x2 y2'.split(' ')
OUT_DPI = 72.0

def convert_figure_info(gold_fig_info):
    assert gold_fig_info['page'] > 0
    in_dpi = gold_fig_info['dpi']
    coords_factor = OUT_DPI / in_dpi
    return dict(caption=gold_fig_info['caption'],
                name=gold_fig_info['name'],
                regionBoundary=dict(zip(BB_KEYS, numpy.array(gold_fig_info['region_bb']) * coords_factor)),
                captionBoundary=dict(zip(BB_KEYS, numpy.array(gold_fig_info['caption_bb']) * coords_factor)),
                page=gold_fig_info['page'] - 1,
                renderDpi=OUT_DPI,
                figType=gold_fig_info['figure_type'])


def convert_gold_pdffigures2_markup(pdfs_dir, out_dir, annotations_file):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    for doc_id, doc_info in annotations.items():
        with open(os.path.join(out_dir, doc_id + '.json'), 'w') as f:
            new_doc_info = [convert_figure_info(fig) for fig in doc_info['figures']]
            json.dump(new_doc_info, f, indent=4)


if __name__ == '__main__':
    convert_gold_pdffigures2_markup(*sys.argv[1:])
