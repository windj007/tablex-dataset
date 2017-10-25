#!/usr/bin/env python

import lxml, lxml.etree, glob, base64, os, subprocess, numpy, io, os, re
from PIL import Image, ImageOps, ImageFilter, ImageDraw

REMOVE_LINES = True
STROKE_RE = re.compile(r'stroke\s*:\s*#([0-9a-f]{6})', re.I)
FILL_RE = re.compile(r'fill\s*:\s*([^;]+)[;$]', re.I)
FILL_OPACITY_RE = re.compile(r'fill-opacity\s*:\s*([^;]+)[;$]', re.I)

def get_stroke(node):
    stroke = node.get('stroke', None)
    if not stroke is None:
        return stroke.strip(' #').lower()
    else:
        style = node.get('style', '')
        stroke = STROKE_RE.search(style)
        return stroke.group(1).lower() if stroke else '000000'


def set_stroke(node, color):
    if not node.get('stroke', None) is None:
        node.set('stroke', '#{}'.format(color))
    else:
        style = node.get('style', '')
        stroke = STROKE_RE.search(style)
        new_stroke = 'stroke:#{};'.format(color)
        if stroke:
            new_style = STROKE_RE.sub(new_stroke, style)
        else:
            new_style = style + ('' if style.endswith(';') else ';') + new_stroke
        node.set('style', new_style)


def set_fill(node, color):
    if not node.get('fill', None) is None:
        node.set('fill', '#{}'.format(color))
    else:
        style = node.get('style', '')
        fill = FILL_RE.search(style)
        new_fill = 'fill:#{};'.format(color)
        if fill:
            new_style = FILL_RE.sub(new_fill, style)
        else:
            new_style = style + ('' if style.endswith(';') else ';') + new_fill
        node.set('style', new_style)


def set_fill_opacity(node, value):
    if not node.get('fill-opacity', None) is None:
        node.set('fill-opacity', value)
    else:
        style = node.get('style', '')
        fill_opacity = FILL_OPACITY_RE.search(style)
        new_fill_opacity = 'fill-opacity:{};'.format(value)
        if fill_opacity:
            new_style = FILL_OPACITY_RE.sub(new_fill_opacity, style)
        else:
            new_style = style + ('' if style.endswith(';') else ';') + new_fill_opacity
        node.set('style', new_style)


FILL_MAPPING = (
    (re.compile('ff[0-4]{4}'), 'ff0000'),
    (re.compile('[0-4]{2}ff[0-4]{2}'), 'ff0000'),
    (re.compile('000000|black'), '00ff00'),
)
def map_fill(stroke):
    for regex, result in FILL_MAPPING:
        if regex.search(stroke):
            return result
    return stroke


def convert_svg(page_with_markup_file, out_dir):
    fname = os.path.splitext(os.path.basename(page_with_markup_file))[0]

    with open(page_with_markup_file) as f:
        src_tree = lxml.etree.parse(f)

    svg = src_tree.getroot()
    image_element = next(iter(n for n in svg.getchildren() if n.tag.endswith('image')))
    dataurl = image_element.get('{http://www.w3.org/1999/xlink}href')
    png_content = base64.b64decode(dataurl.rsplit(',', 1)[1])
    in_fname = os.path.join(out_dir, '{}_in.png'.format(fname))
    with open(in_fname, 'wb') as f:
        f.write(png_content)

    svg_fname = os.path.join(out_dir, '{}_out.svg'.format(fname))
    out_fname = os.path.join(out_dir, '{}_out.png'.format(fname))

    if REMOVE_LINES:
        for n in svg.getchildren():
            if n.tag.endswith('path'):
                svg.remove(n)

    for n in list(svg.getchildren()):
        if n.tag.endswith('rect'):
            stroke = get_stroke(n)
            new_fill = map_fill(stroke)
            set_fill(n, new_fill)
            set_stroke(n, new_fill)
            set_fill_opacity(n, '1')

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
    # os.remove(svg_fname)

    out_img = Image.open(out_fname).convert('RGB')
    out_img.save(out_fname)
    return in_fname, out_fname


def make_inout_pairs(in_dir='/notebook/data/3_prepared_images/', out_dir='/notebook/data/4_inout_pairs/'):
    for page_with_markup_file in glob.glob(os.path.join(in_dir, '*.svg')):
        convert_svg(page_with_markup_file, out_dir)


if __name__ == '__main__':
    make_inout_pairs()
