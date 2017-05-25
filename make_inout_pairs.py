#!/usr/bin/env python

import lxml, lxml.etree, glob, base64, os, subprocess, numpy, io, os
from PIL import Image, ImageOps, ImageFilter, ImageDraw

for page_with_markup_file in glob.glob('/notebook/data/3_prepared_images/*.svg'):
    fname = os.path.splitext(os.path.basename(page_with_markup_file))[0]

    with open(page_with_markup_file) as f:
        src_tree = lxml.etree.parse(f)

    svg = src_tree.getroot()
    image_element = next(iter(n for n in svg.getchildren() if n.tag.endswith('image')))
    dataurl = image_element.get('{http://www.w3.org/1999/xlink}href')
    png_content = base64.b64decode(dataurl.rsplit(',', 1)[1])
    with open('/notebook/data/4_inout_pairs/{}_in.png'.format(fname), 'wb') as f:
        f.write(png_content)

    svg_fname = '/notebook/data/4_inout_pairs/{}_out.svg'.format(fname)
    out_fname = '/notebook/data/4_inout_pairs/{}_out.png'.format(fname)

    size = (int(svg.get('width')), int(svg.get('height')))
    bg_image = Image.fromarray(numpy.ones(size),
                                       mode='1')
    buf = io.BytesIO()
    bg_image.save(buf, format='png')
    image_element.set('{http://www.w3.org/1999/xlink}href',
                      base64.b64encode(buf.getvalue()))
    with open(svg_fname, 'w') as f:
        f.write(lxml.etree.tostring(src_tree).decode('utf8'))
    subprocess.run(['convert', svg_fname, out_fname])
    os.remove(svg_fname)

    out_img = Image.open(out_fname).convert('RGB')
    out_img = ImageOps.autocontrast(out_img)
    ImageDraw.floodfill(out_img, (0, 0), 0)
    out_img = out_img.filter(ImageFilter.GaussianBlur(3))
    out_img.save(out_fname)
