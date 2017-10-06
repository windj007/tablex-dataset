import subprocess, os, glob, tempfile, sys, TexSoup, pdfquery, \
    re, collections, numpy, cv2, json, itertools, joblib, \
    shutil, random
from pdfquery.cache import FileCache
from PIL import Image
from IPython.display import display
from timeout_decorator import timeout

from prepare_images_utils import *


DENSITY=100
PIXELS_TO_POINTS_FACTOR = 72.0 / DENSITY
POINTS_TO_PIXELS_FACTOR = DENSITY / 72.0

def pdf_to_pages(in_file, out_dir, pages=None):
    if pages is None:
        subprocess.check_call(['convert',
                               '-define', 'pdf:use-cropbox=false',
                               '-density', str(DENSITY),
                               in_file,
                               '-sharpen', '0x1.0',
#                                '-resample', '{0}x{0}'.format(DENSITY),
                               os.path.join(out_dir, '%04d.png')])
    else:
        for page in pages:
            subprocess.check_call(['convert',
                                   '-define', 'pdf:use-cropbox=false',
                                   '-density', str(DENSITY),
                                   '{}[{}]'.format(in_file, page),
                                   '-sharpen', '0x1.0',
#                                    '-resample', '{0}x{0}'.format(DENSITY),
                                   os.path.join(out_dir, '{:04d}.png'.format(page))])
    result = list(glob.glob(os.path.join(out_dir, '*.png')))
    result.sort()
    return result


def boxes_to_mask(page_image, boxes):
    demo_mask = numpy.zeros((page_image.size[1], page_image.size[0], 3),
                            dtype='uint8')
    for channel, box in boxes:
        y1, x1, y2, x2 = box
        color = [0] * 3
        color[channel] = 255
        cv2.drawContours(demo_mask,
                         [numpy.array([(x1, y1),
                                       (x2, y1),
                                       (x2, y2),
                                       (x1, y2)])],
                         -1,
                         tuple(color),
                         cv2.FILLED)
    return arr_to_img(demo_mask.astype('float32') / 255.0)


def make_demo_mask(page_image, boxes):
    demo_mask = boxes_to_mask(page_image, boxes)
    demo_mask_blended = Image.blend(page_image.convert('RGB'), demo_mask, 0.5)
    return demo_mask_blended


def guess_main_latex_file(project_dir):
    latex_files_in_wd = list(glob.glob(os.path.join(project_dir, '*.tex')))
    if len(latex_files_in_wd) == 0:
        raise Exception('No latex files in folder!')
    elif len(latex_files_in_wd) > 1:
        raise Exception('Many latex files in folder: {}'.format(latex_files_in_wd))
    return latex_files_in_wd[0]


def compile_latex(project_dir):
    our_latex_file = guess_main_latex_file(project_dir)
    our_latex_filename = os.path.basename(our_latex_file)
    subprocess.check_call(['pdflatex', '-synctex=1', '-interaction=nonstopmode', our_latex_filename],
                          cwd=project_dir)
    return our_latex_file


ARCHIVE_SUFFIX = '.tar.gz'
def read_metadata(archive):
    with open(archive[:-len(ARCHIVE_SUFFIX)] + '.js', 'r') as f:
        return json.load(f)


def _split_line_by_caption(line):
    start = line.find(r'\caption')
    if start < 0:
        return line
    if line.find('%', 0, start) >= 0:
        return line
    return line[:start] + '\n' + line[start:]


LINE_COMMENT_RE = re.compile(r'^(.*?)(?<!\\)(%.*$)')
def _truncate_comments(line):
    return LINE_COMMENT_RE.sub(r'\1%', line)


def preprocess_latex_content(content):
    content = '\n'.join(_truncate_comments(line)
                        for line in content.split('\n'))
    content = '\n'.join(_split_line_by_caption(line)
                        for line in content.split('\n'))
    return content


def preprocess_latex_file(fname):
    with open(fname, 'r') as f:
        content = f.read()
    content = preprocess_latex_content(content)
    with open(fname, 'w') as f:
        f.write(content)


def contains_something_interesting(archive, positions_getter):
    meta = read_metadata(archive)
    if meta['content_type'] != 'application/x-eprint-tar':
        return False

    try:
        with tempfile.TemporaryDirectory() as wd:
            subprocess.check_call(['tar', 'xfv', archive, '-C', wd])
            latex_file = guess_main_latex_file(wd)

            with open(latex_file, 'r') as f:
                parse_tree = TexSoup.TexSoup(f.read())

            try:
                next(positions_getter(parse_tree))
            except StopIteration:
                return False
            return True
    except Exception as ex:
#         print('Could not handle {}: {}'.format(archive, ex))
        return False


SYNCTEX_OUTPUT_BOX_RE = re.compile(r'''
Output:(?P<output>.*?)\s+
Page:(?P<page>.*?)\s+
x:(?P<x>.*?)\s+
y:(?P<y>.*?)\s+
h:(?P<h>.*?)\s+
v:(?P<v>.*?)\s+
W:(?P<W>.*?)\s+
H:(?P<H>.*?)\s+
before:(?P<before>.*?)\s+
offset:(?P<offset>.*?)\s+
middle:(?P<middle>.*?)\s+
after:(?P<after>.*?)\s+''', re.X)

def parse_synctex_output(output):
    result = collections.defaultdict(list)
    for m in SYNCTEX_OUTPUT_BOX_RE.finditer(output):
        bb_v = float(m.group('v'))
        bb_h = float(m.group('h'))
        bb_y = float(m.group('y'))
        bb_width = float(m.group('W'))
        bb_height = float(m.group('H'))
        depth = bb_v - bb_y
        result[int(m.group('page'))].append([bb_y - bb_height, bb_h, bb_y + depth, bb_h + bb_width])
    return result


def no_bbox_aggregation(pdf, page, boxes, tokens):
    return boxes


def bbox_union(pdf, page, boxes, tokens):
    boxes = list(boxes)
    if len(boxes) == 0:
        return []
    y1s, x1s, y2s, x2s = zip(*boxes)
    return [[min(y1s), min(x1s), max(y2s), max(x2s)]]


def convert_coords_to_pq(box, cropbox):
    ul_y, ul_x, br_y, br_x = box
    x_off, _, _, page_height = cropbox
    return numpy.array([ul_x + x_off, page_height - br_y, br_x + x_off, page_height - ul_y])


def convert_coords_from_pq(box, cropbox):
    bl_x, bl_y, ur_x, ur_y = box
    x_off, _, _, page_height = cropbox
    return numpy.array([page_height - ur_y, bl_x - x_off, page_height - bl_y, ur_x - x_off])


FLOAT_RE = re.compile(r'(\d+(\.\d+)?)')
def get_pg_elem_bbox(elem):
    return [float(t[0]) for t in FLOAT_RE.findall(elem.attrib['bbox'])]


def box_is_good(src, found, max_height_ratio=1.99, max_width_ratio=1.99, min_dice = 0.3):
    sy1, sx1, sy2, sx2 = src
    fy1, fx1, fy2, fx2 = found

    # if found is totally inside src, then it is good
    if fy1 >= sy1 and fx1 >= sx1 and fy2 <= sy2 and fx2 <= sx2:
        return True

    sh, sw = (sy2 - sy1), (sx2 - sx1)
    fh, fw = (fy2 - fy1), (fx2 - fx1)

    if fh / sh > max_height_ratio:
        return False

    if fw / sw > max_width_ratio:
        return False

    sa = sw * sh
    fa = fw * fh

    iy1, ix1, iy2, ix2 = max(sy1, fy1), max(sx1, fx1), min(sy2, fy2), min(sx2, fx2)
    ia = (iy2 - iy1) * (ix2 - ix1)
    dice = ia / (sa + fa - ia)
    return dice >= min_dice


PQ_IN_BBOX = 'LTPage[pageid="{page}"] LTTextLineHorizontal:overlaps_bbox("{bbox}"), LTPage[pageid="{page}"] LTTextLineVertical:overlaps_bbox("{bbox}")'
# PQ_IN_BBOX = 'LTPage[pageid="{page}"] LTTextLineHorizontal:in_bbox("{bbox}")'
PQ_BBOX_PAD = numpy.array([-6, -6, 6, 6])
def aggregate_object_bboxes(pdf, page, boxes, tokens, union=True):
    cb = pdf.get_page(page).cropbox
    good_boxes = []
    for src_box in boxes:
        overlapping_elems = pdf.pq(
            PQ_IN_BBOX.format(page=page,
                              bbox=', '.join(map(str,
                                                 convert_coords_to_pq(numpy.array(src_box) + PQ_BBOX_PAD,
                                                                      cb)))))
        for found_elem in overlapping_elems:
            found_box = convert_coords_from_pq(get_pg_elem_bbox(found_elem), cb)
            if box_is_good(src_box, found_box):
                good_boxes.append(found_box)

    if union:
        return bbox_union(pdf,
                          page,
                          good_boxes,
                          tokens)
    else:
        return good_boxes


def aggregate_object_bboxes_no_union(pdf, page, boxes, tokens):
    return aggregate_object_bboxes(pdf, page, boxes, tokens, union=False)


@timeout(120)
def parse_pdf(fname):
    pdf = pdfquery.PDFQuery(fname)
    pdf.load()
    return pdf


def pdf2samples(archive, out_dir, synctex_positions_getter, boxes_aggregator=bbox_union, display_demo=False):
    try:
        meta = read_metadata(archive)
        if meta['content_type'] != 'application/x-eprint-tar':
            return None
        paper_id = os.path.basename(archive)[:len('.tar.gz')]

        with tempfile.TemporaryDirectory() as wd:
            subprocess.check_call(['tar', 'xfv', archive, '-C', wd])

            our_latex_file = guess_main_latex_file(wd)
            our_latex_filename = os.path.basename(our_latex_file)
            target_pdf_file = os.path.join(wd, os.path.splitext(our_latex_filename)[0] + '.pdf')
            target_pdf_filename = os.path.basename(target_pdf_file)

            preprocess_latex_file(our_latex_file)
            try:
                compile_latex(wd)
            except subprocess.CalledProcessError as ex:
                src_log_file = os.path.splitext(our_latex_file)[0] + '.log'
                shutil.copy2(src_log_file,
                             os.path.join(out_dir,
                                          '{}.log'.format(paper_id)))
                raise

            with open(our_latex_file, 'r') as f:
                parse_tree = TexSoup.TexSoup(f.read())

            pdf = parse_pdf(target_pdf_file)

            with tempfile.TemporaryDirectory(dir=wd) as pages_wd:
                page_files = pdf_to_pages(target_pdf_file, pages_wd)

                segments_by_page = collections.defaultdict(list)
                objects = synctex_positions_getter(parse_tree)
                for category, obj, tokens in objects:
                    obj_boxes_by_page = collections.defaultdict(list)
                    for line_to_show, char_to_show in obj:
                        synctex_output = subprocess.check_output(['synctex', 'view',
                                                                  '-i', '{}:{}:{}'.format(line_to_show + 1,
                                                                                          char_to_show + 1,
                                                                                          our_latex_filename),
                                                                  '-o', target_pdf_filename],
                                                                 cwd=wd).decode('ascii')
                        for page, boxes in parse_synctex_output(synctex_output).items():
                            obj_boxes_by_page[page].extend(boxes)

                    for page, boxes in obj_boxes_by_page.items():
                        for box in boxes_aggregator(pdf, page, boxes, tokens):
                            segments_by_page[page].append((category, box))

                for page, segments_with_categories in segments_by_page.items():
                    markup = [(cat,
                               (numpy.array(box) * POINTS_TO_PIXELS_FACTOR).astype('int'))
                              for cat, box in segments_with_categories]

                    page_img = load_image_opaque(page_files[page - 1])
                    
                    shutil.copy2(page_files[page - 1],
                                 os.path.join(out_dir,
                                              '{}_{:04d}_in.png'.format(paper_id, page)))
                    out_img = boxes_to_mask(page_img, markup)
                    out_img.save(os.path.join(out_dir,
                                              '{}_{:04d}_out.png'.format(paper_id, page)))
                    mask = make_demo_mask(page_img, markup)
                    mask.save(os.path.join(out_dir,
                                           '{}_{:04d}_demo.png'.format(paper_id, page)))
                    if display_demo:
                        display(mask)
    except TimeoutError:
        print('pdfquery took too long {}'.format(archive))


def get_maketitle(soup):
    return [(1,
             [soup.char_pos_to_line(soup.maketitle.name.position)],
             [''])]


def get_all_tokens(root, include_command_names=True, tokenize=True):
    if include_command_names and isinstance(root, TexSoup.data.TexNode):
        if isinstance(root.name, TexSoup.TokenWithPosition):
            yield root.name
    for ch in root.contents:
        if isinstance(ch, TexSoup.TokenWithPosition):
            if tokenize:
                for tok in ch.split(' '):
                    if tok:
                        yield tok
            else:
                yield ch
        elif isinstance(root, TexSoup.data.TexNode):
            yield from get_all_tokens(ch, include_command_names=include_command_names)


def get_table_info(soup):
    for table in list(soup.find_all('table')):
        caption = table.caption
        if not caption is None:
            yield (0,
                   [soup.char_pos_to_line(tok.position)
                    for tok in get_all_tokens(caption)],
                   [tok.text for tok in get_all_tokens(caption, False)])
        tabular = table.tabular
        if not tabular is None:
            yield (1,
                   [soup.char_pos_to_line(tok.position)
                    for tok in get_all_tokens(tabular)],
                   [tok.text for tok in get_all_tokens(tabular, False)])
