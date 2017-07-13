#!/usr/bin/env python

import lxml, lxml.etree, glob, base64, os, subprocess, numpy, io, os, re
from PIL import Image, ImageOps, ImageFilter, ImageDraw

REMOVE_LINES = True
STROKE_RE = re.compile(r'stroke\s*:\s*#([0-9a-f]{6})', re.I)
FILL_RE = re.compile(r'fill\s*:\s*([^;]+)[;$]', re.I)
FILL_OPACITY_RE = re.compile(r'fill-opacity\s*:\s*([^;]+)[;$]', re.I)

def get_stroke(style):
    stroke = STROKE_RE.search(style)
    return stroke.group(1) if stroke else '000000'


def set_fill(style, color):
    fill = FILL_RE.search(style)
    new_fill = 'fill:#{};'.format(color)
    if fill:
        return FILL_RE.sub(new_fill, style)
    else:
        return style + ('' if style.endswith(';') else ';') + new_fill


def set_fill_opacity(style, value):
    fill_opacity = FILL_OPACITY_RE.search(style)
    new_fill_opacity = 'fill-opacity:{};'.format(value)
    if fill_opacity:
        return FILL_OPACITY_RE.sub(new_fill_opacity, style)
    else:
        return style + ('' if style.endswith(';') else ';') + new_fill_opacity


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

    if REMOVE_LINES:
        for n in svg.getchildren():
            if n.tag.endswith('path'):
                svg.remove(n)

    for n in list(svg.getchildren()):
        if n.tag.endswith('rect'):
            style = n.get('style') or ''
            stroke = get_stroke(style)
            if stroke != '000000':
                n.set('style', set_fill(style, stroke))
            elif stroke == '000000':
                n.set('style', set_fill(style, 'ffffff'))
            n.set('style', set_fill_opacity(n.get('style'), '1'))

    size = (int(svg.get('height')), int(svg.get('width')))
    bg_image = Image.fromarray(numpy.zeros(size, dtype='uint8'),
                               mode='L')
    svg.remove(image_element)
    svg.insert(0, lxml.etree.Element("rect",
                                     dict(style="fill:black;stroke:black;stroke-width:0",
                                          id="rect000",
                                          x='0',
                                          y='0',
                                          height=str(size[0]),
                                          width=str(size[1]))))

    with open(svg_fname, 'w') as f:
        f.write(lxml.etree.tostring(src_tree).decode('utf8'))
    subprocess.run(['convert', svg_fname, '-antialias', out_fname])
    os.remove(svg_fname)

    out_img = Image.open(out_fname).convert('RGB')
    out_img.save(out_fname)
