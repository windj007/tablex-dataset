#!/usr/bin/env python

import glob, json, os, base64, svgwrite
from PIL import Image

DPI = 72

tables_info = {}
for pdffig_res_file in glob.glob('/notebook/data/1_pdffigures2_out/*.json'):
    file_id = os.path.splitext(os.path.basename(pdffig_res_file))[0]
    with open(pdffig_res_file, 'r') as f:
        pdffig_res = json.load(f)
    for obj_info in pdffig_res:
        if obj_info['figType'] == 'Table':
            tables_info[(file_id, obj_info['page'])] = obj_info


for page_img_fname in glob.glob('/notebook/data/2_page_images/*.png'):
    basename = os.path.splitext(os.path.basename(page_img_fname))[0]
    pdf_id, page_no = basename.split('-')
    page_no = int(page_no)
    cur_table_info = tables_info.get((pdf_id, page_no), None)

    with open(page_img_fname, 'rb') as f:
        dataurl = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
    image_size = Image.open(page_img_fname).size

    dwg = svgwrite.Drawing('/notebook/data/3_prepared_images/{}.svg'.format(basename),
                           size=image_size,
                           profile='tiny')
    dwg.add(dwg.image(dataurl))
    
    if not cur_table_info is None:
        rect = cur_table_info['regionBoundary']
        dwg.add(dwg.rect((rect['x1'], rect['y1']),
                         (rect['x2'] - rect['x1'],
                          rect['y2'] - rect['y1']),
                         **{'fill-opacity' : 0.0,
                            'stroke' : 'black',
                            'stroke-width' : 2,
                            'stroke-opacity' : 1 }))

    dwg.save()
