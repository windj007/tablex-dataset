import subprocess, os, glob, tempfile, sys, TexSoup, pdfquery, \
    re, collections, numpy, cv2, json, itertools, joblib, \
    shutil, random, traceback, html, ngram, scipy.optimize, \
    scipy.spatial, rtree, difflib
from pdfquery.cache import FileCache
from PIL import Image
from IPython.display import display
from timeout_decorator import timeout
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar, LTFigure, LTAnno

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
#     for f in result:
#         display(Image.open(f))
    return result


CLS_TO_COLOR = [
    [255, 0, 0], # caption
    [0, 255, 0], # body
    [0, 0, 255], # cells
    [255, 255, 0], # rows
    [255, 0, 255], # cols
    [255, 128, 0],
    [255, 0, 128],
    [128, 255, 0],
    [128, 0, 255]
]
def boxes_to_mask(page_image, boxes):
    demo_mask = numpy.zeros((page_image.size[1], page_image.size[0], 3),
                            dtype='uint8')
    for cls, box in boxes:
        y1, x1, y2, x2 = box

        cv2.drawContours(demo_mask,
                         [numpy.array([(x1, y1),
                                       (x2, y1),
                                       (x2, y2),
                                       (x1, y2)]).astype('int')],
                         -1,
                         tuple(CLS_TO_COLOR[cls]),
                         # cv2.FILLED
                        )
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
    target_pdf_file = os.path.splitext(our_latex_file)[0] + '.pdf'
    exception = None
    try:
        subprocess.check_call(['latexmk',
                               '-pdf',
                               '-f',
                               '-shell-escape',
                               '-synctex=1',
                               '-interaction=nonstopmode',
                               our_latex_filename],
                              cwd=project_dir)
    except subprocess.CalledProcessError as ex:
        try:
            subprocess.check_call(['pdflatex',
                                   '-shell-escape',
                                   '-synctex=1',
                                   '-interaction=nonstopmode',
                                   our_latex_filename],
                                  cwd=project_dir)
        except subprocess.CalledProcessError as ex:
            exception = ex
    if os.path.exists(target_pdf_file):
        return our_latex_file
    else:
        raise exception


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


USE_PACKAGE_RE = re.compile(r'^\\usepackage.*$', re.M)
PACKAGES_TO_ADD = r'\usepackage{auto-pst-pdf}'
def _inject_packages(src):
    first_usepackage = USE_PACKAGE_RE.search(src)
    if not first_usepackage:
        return src
    return src[:first_usepackage.end(0)] + '\n' + PACKAGES_TO_ADD + src[first_usepackage.end(0):]


def preprocess_latex_content(content):
    content = '\n'.join(_truncate_comments(line)
                        for line in content.split('\n'))
    content = '\n'.join(_split_line_by_caption(line)
                        for line in content.split('\n'))
    content = _inject_packages(content)
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


def map_to_pdf_coords(line_to_show, char_to_show, latex, pdf, wd):
    synctex_output = subprocess.check_output(['synctex', 'view',
                                              '-i', '{}:{}:{}'.format(line_to_show + 1,
                                                                      char_to_show + 1,
                                                                      latex),
                                              '-o', pdf],
                                             cwd=wd).decode('ascii')
    return parse_synctex_output(synctex_output)


def box_area(b):
    x1, y1, x2, y2 = b
    w, h = max(x2 - x1, 0), max(y2 - y1, 0)
    return w * h


def box_inter_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = (max(ax1, bx1),
                          max(ay1, by1),
                          min(ax2, bx2),
                          min(ay2, by2))
    res = box_area((ix1, iy1, ix2, iy2))
#     print('box_inter_area', a, b, (ix1, iy1, ix2, iy2), res)
    return res


def intersects_any(target, golds):
    return any(box_inter_area(target, g) > 0 for g in golds)


def included(what, to):
    ax1, ay1, ax2, ay2 = what
    bx1, by1, bx2, by2 = to
    return ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2


def included_into_any(what, golds):
    return any(included(what, g) for g in golds)


def get_box_center(box):
    y1, x1, y2, x2 = box
    return numpy.array([(y1 + y2) / 2, (x1 + x2) / 2])


def is_point_in_box(point, box):
    y, x = point
    y1, x1, y2, x2 = box
    return y >= y1 and x >= x1 and y <= y2 and x <= x2


def center_in_any(what, golds):
    center = get_box_center(what)
    return any(is_point_in_box(center, g) for g in golds)


# Initially taken from http://zderadicka.eu/parsing-pdf-for-fun-and-profit-indeed-in-python/ and then modified
class PdfMinerWrapper(object):
    def __init__(self, pdf_doc, pdf_pwd=""):
        self.pdf_doc = pdf_doc
        self.pdf_pwd = pdf_pwd
        self.loaded = False
        self.index_by_page = {}
        self.page_cache = {}
        self.device = self.interpreter = self.pages = None

    def load(self):
        if self.loaded:
            return
        self.fp = open(self.pdf_doc, 'rb')
        parser = PDFParser(self.fp)
        doc = PDFDocument(parser, password=self.pdf_pwd)
        parser.set_document(doc)
        self.doc = doc
        self.index_by_page = {}
        self.page_cache = {}

        rsrcmgr = PDFResourceManager()
        self.device = PDFPageAggregator(rsrcmgr,
                                        laparams=LAParams(char_margin=0.1,
                                                          all_texts=True))
        self.interpreter = PDFPageInterpreter(rsrcmgr, self.device)
        self.pages = list(PDFPage.create_pages(self.doc))

        self.loaded = True

    def close(self):
        if not self.loaded:
            return
        self.fp.close()
        self.index_by_page = {}
        self.page_cache = {}
        del self.device
        del self.interpreter
        del self.pages
        self.device = self.interpreter = self.pages = None
        self.loaded = False

    def get_page(self, page_no):
        '''Returns: LTPage'''
        if not page_no in self.page_cache:
            page = self.pages[page_no]
            self.interpreter.process_page(page)
            self.page_cache[page_no] = (self.device.get_result(), page)
        return self.page_cache[page_no]

    def get_boxes(self, page_no, gold_bboxes):
        return self.__get_boxes(self.get_page(page_no)[0], gold_bboxes)

    def get_text(self, page_no, gold_bboxes, glue=''):
        elems = list(self.get_boxes(page_no, gold_bboxes))
        elems.sort(key=lambda b: (-b.bbox[1], b.bbox[0]))
        return glue.join(e.get_text() for e in elems)

    def __get_boxes(self, root, gold_boxes):
        if isinstance(root, LTAnno) or not intersects_any(root.bbox, gold_boxes):
            return
        if isinstance(root, (LTChar,)) \
            and center_in_any(root.bbox, gold_boxes):
            yield root
            return
        if not isinstance(root, LTChar):
            try:
                iter(root)
            except TypeError:
                return
            for ch in root:
                for b in self.__get_boxes(ch, gold_boxes):
                    yield b

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, _type, value, traceback):
        self.close()


def no_bbox_aggregation(pdf, page, boxes, tokens):
    return boxes


PQ_BBOX_PAD = numpy.array([-1, -1, 1, 1])

def bbox_union(pdf, page, boxes, tokens):
    boxes = list(boxes)
    if len(boxes) == 0:
        return []
    cropbox = pdf.get_page(page - 1)[1].cropbox
    texts = [pdf.get_text(page - 1,
                          [convert_coords_to_pq(box, cropbox) + PQ_BBOX_PAD])
             for box in boxes]
    y1s, x1s, y2s, x2s = zip(*boxes)
    result = (min(y1s), min(x1s), max(y2s), max(x2s))
    return [(result, list(zip(boxes, texts)))]


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


def aggregate_object_bboxes(pdf, page, boxes, tokens, union=True):
    cropbox = pdf.get_page(page - 1)[1].cropbox
    overlapping_elems = list(pdf.get_boxes(page - 1,
                                           [convert_coords_to_pq(box, cropbox) + PQ_BBOX_PAD
                                            for box in boxes]))
    good_boxes = [convert_coords_from_pq(found_elem.bbox, cropbox)
                  for found_elem in overlapping_elems]
    if union and good_boxes:
        return bbox_union(pdf,
                          page,
                          good_boxes,
                          tokens)
    else:
        return [(b,
                 [b,
                  pdf.get_text(page - 1,
                               [convert_coords_to_pq(b, cropbox) + PQ_BBOX_PAD])])
                for b in good_boxes]


def aggregate_object_bboxes_no_union(pdf, page, boxes, tokens):
    return aggregate_object_bboxes(pdf, page, boxes, tokens, union=False)


@timeout(300)
def parse_pdf(fname):
    pdf = PdfMinerWrapper(fname)
    pdf.load()
    return pdf


def partition_two_rectangles(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = (max(ax1, bx1),
                          max(ay1, by1),
                          min(ax2, bx2),
                          min(ay2, by2))
    i_area = box_area((ix1, iy1, ix2, iy2))
    if i_area < 1e-4:
        return [a, b]
    result = []
    sorted_x = sorted({ax1, ax2, bx1, bx2})
    sorted_y = sorted({ay1, ay2, by1, by2})
    for xi in range(len(sorted_x) - 1):
        for yi in range(len(sorted_y) - 1):
            cand = (sorted_x[xi],
                    sorted_y[yi],
                    sorted_x[xi+1],
                    sorted_y[yi+1])
            if box_inter_area(cand, a) > 1e-4 or box_inter_area(cand, b) > 1e-4:
                result.append(cand)
    return result


def partition_many_rectangles(boxes):
    result = { i : b for i, b in enumerate(boxes) }
    idx = rtree.index.Rtree(interleaved=True)
    for i, b in result.items():
        idx.insert(i, b)
    next_id = len(result)
    something_new = True
    while something_new:
        something_new = False
        intersects_with_any = False
        for cur_i, cur_b in result.items():
            intersects_with_any = False
            for other_i in idx.intersection(cur_b):
                if cur_i == other_i or not other_i in result:
                    continue
                other_b = result[other_i]
                if box_inter_area(cur_b, other_b) < 1:
                    continue
                intersects_with_any = True
                break
            if intersects_with_any:
                break

        if intersects_with_any:
            del result[cur_i]
            del result[other_i]
            something_new = True
            for new_b in partition_two_rectangles(cur_b, other_b):
                result[next_id] = new_b
                idx.insert(next_id, new_b)
                next_id += 1
    return list(result.values())


def json_encoder_default_for_numpy(o):
    if isinstance(o, numpy.ndarray):
        return o.tolist()
    raise TypeError()


def pdf_latex_to_samples(paper_id,
                         wd,
                         our_latex_file,
                         target_pdf_file,
                         out_dir,
                         synctex_positions_getter,
                         boxes_aggregator=bbox_union,
                         display_demo=False):
    with open(our_latex_file, 'r') as f:
        parse_tree = TexSoup.TexSoup(f.read())

    pdf = parse_pdf(target_pdf_file)

    with tempfile.TemporaryDirectory(dir=wd) as pages_wd:
        page_files = pdf_to_pages(target_pdf_file, pages_wd)

        segments_by_page = collections.defaultdict(list)
        objects = list(synctex_positions_getter(parse_tree))
        objects.sort(key=lambda t: t[0])
        for category, is_atomic, obj, tokens, groupings in objects:
            if is_atomic:
                obj_boxes_by_page = collections.defaultdict(list)
                for line_to_show, char_to_show in obj:
                    for page, boxes in map_to_pdf_coords(line_to_show,
                                                         char_to_show,
                                                         our_latex_file,
                                                         target_pdf_file,
                                                         wd).items():
                        obj_boxes_by_page[page].extend(boxes)

                for page, boxes in obj_boxes_by_page.items():
                    cropbox = pdf.get_page(page - 1)[1].cropbox
                    for box, atom_boxes_with_texts in boxes_aggregator(pdf, page, boxes, tokens):
                        segments_by_page[page].append((category,
                                                       box,
                                                       atom_boxes_with_texts))
            else:
                # in this case obj and tokens are lists of lists
                obj_boxes_by_page = collections.defaultdict(list)
                for sub_obj in obj:
                    for line_to_show, char_to_show in sub_obj:
                        for page, boxes in map_to_pdf_coords(line_to_show,
                                                             char_to_show,
                                                             our_latex_file,
                                                             target_pdf_file,
                                                             wd).items():
                            obj_boxes_by_page[page].extend(boxes)
                assert len(obj_boxes_by_page) == 1
                page, boxes = next(iter(obj_boxes_by_page.items()))
                boxes = partition_many_rectangles(boxes)
                assigned_boxes = reassign_boxes_greedy(pdf, page, boxes, tokens)
                for part_boxes, part_tokens in zip(assigned_boxes, tokens):
                    for box, atom_boxes_with_texts in boxes_aggregator(pdf, page, part_boxes, part_tokens):
                        segments_by_page[page].append((category,
                                                       box,
                                                       atom_boxes_with_texts))
                for group_category, group_elems in groupings:
                    group_boxes = [box
                                   for elem_i in group_elems
                                   for box in assigned_boxes[elem_i]]
                    group_tokens = [tok
                                    for elem_i in group_elems
                                    for tok_list in tokens
                                    for tok in tok_list]
                    for box, atom_boxes_with_texts in boxes_aggregator(pdf, page, group_boxes, group_tokens):
                        segments_by_page[page].append((group_category,
                                                       box,
                                                       atom_boxes_with_texts))

        shutil.copy2(our_latex_file,
                     os.path.join(out_dir,
                                  '{}.tex'.format(paper_id)))
        for page, segments_with_categories in segments_by_page.items():
            markup = [(cat,
                       (numpy.array(box) * POINTS_TO_PIXELS_FACTOR).astype('int'))
                      for cat, box, atom_boxes_with_texts in segments_with_categories]

            page_img = load_image_opaque(page_files[page - 1])

            shutil.copy2(page_files[page - 1],
                         os.path.join(out_dir,
                                      '{}_{:04d}_in.png'.format(paper_id, page)))
            out_img = boxes_to_mask(page_img, markup)
            out_img.save(os.path.join(out_dir,
                                      '{}_{:04d}_out.png'.format(paper_id, page)))
            with open(os.path.join(out_dir,
                                   '{}_{:04d}_out.json'.format(paper_id, page)), 'w') as f:
                json.dump(segments_with_categories,
                          f,
                          default=json_encoder_default_for_numpy)
            mask = make_demo_mask(page_img, markup)
            mask.save(os.path.join(out_dir,
                                   '{}_{:04d}_demo.png'.format(paper_id, page)))
            if display_demo:
                display(mask)


def pdf2samples(archive, out_dir, synctex_positions_getter, boxes_aggregator=bbox_union, display_demo=False):
    try:
        meta = read_metadata(archive)
        if meta['content_type'] != 'application/x-eprint-tar':
            return None
        paper_id = os.path.basename(archive)[:-len('.tar.gz')]

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

            pdf_latex_to_samples(paper_id, wd,
                                 our_latex_file,
                                 target_pdf_file,
                                 out_dir,
                                 synctex_positions_getter,
                                 boxes_aggregator=boxes_aggregator,
                                 display_demo=display_demo)
    except TimeoutError:
        print('pdfquery took too long {}'.format(archive))


def get_maketitle(soup):
    return [(1,
             [soup.char_pos_to_line(soup.maketitle.name.position)],
             [''])]


GET_ALL_TOKENS_COMMANDS_TO_IGNORE = {'cline'}
GET_ALL_TOKENS_COMMANDS_WITHOUT_NAME = {'hline', 'makecell', 'multirow', 'multicolumn'}
GET_ALL_TOKENS_COMMANDS_CONTENTS_TO_SKIP = { 'multirow' : 2, 'multicolumn' : 2 }
def get_all_tokens(root, include_command_names=True, tokenize=True):
    if isinstance(root, TableCell):
        for ch in root.contents:
            for tok in get_all_tokens(ch, include_command_names=include_command_names, tokenize=tokenize):
                yield tok
    elif isinstance(root, TexSoup.TokenWithPosition):
        if tokenize:
            for tok in root.split(' '):
                if tok:
                    yield tok
        else:
            yield root
    else:
        if isinstance(root, TexSoup.TexNode) and root.name in GET_ALL_TOKENS_COMMANDS_TO_IGNORE:
            return
        if include_command_names and isinstance(root, TexSoup.TexNode):
            if isinstance(root.name, TexSoup.TokenWithPosition) and not root.name.text in GET_ALL_TOKENS_COMMANDS_WITHOUT_NAME:
                yield root.name

        if isinstance(root.name, TexSoup.TokenWithPosition) and root.name.text in GET_ALL_TOKENS_COMMANDS_CONTENTS_TO_SKIP:
            contents_to_skip = GET_ALL_TOKENS_COMMANDS_CONTENTS_TO_SKIP[root.name.text]
        else:
            contents_to_skip = 0

        for ch in list(root.contents)[contents_to_skip:]:
            for tok in get_all_tokens(ch, include_command_names=include_command_names, tokenize=tokenize):
                yield tok


AMP_RE = re.compile(r'(^|[^\\\d])(&)')
NEWLINE_RE = re.compile(r'(^|.)(\\\\)($|\s)')


class TableCell(object):
    def __init__(self, contents=None, colspan=1, rowspan=1):
        self.contents = contents or []
        self.colspan = colspan
        self.rowspan = rowspan

    def to_html(self):
        attribs = ''
        if self.colspan > 1:
            attribs += 'colspan="{}"'.format(self.colspan)
        if self.rowspan > 1:
            attribs += (' ' if attribs else '') + 'rowspan="{}"'.format(self.rowspan)
        return '<td{}>{}</td>'.format((' ' if attribs else '') + attribs,
                                      html.escape(' '.join(map(str, self.contents))))


class Table(object):
    def __init__(self, rows):
        self.rows = rows
    
    def _repr_html_(self):
        return '<table>{}</table>'.format(
            '\n'.join(
                '<tr>{}</tr>'.format(
                    '\n'.join(
                        cell.to_html() for cell in row
                    )
                )
                for row in self.rows
            )
        )


def split_row_into_cells(row):
    result = []
    cur_cell = TableCell()
    for ch in row:
        if isinstance(ch, TexSoup.TokenWithPosition):
            ch = ch.strip()
            start = 0
            while start < len(ch.text):
                new_start_offset = 0
                match = AMP_RE.search(ch.text[start:])
                if match:
                    new_start_offset = match.start(2)
                    token_to_append = ch[start:start+new_start_offset]
                    if len(token_to_append.strip().text) > 0:
                        cur_cell.contents.append(token_to_append)
                    result.append(cur_cell)
                    cur_cell = TableCell()
                else:
                    break
                start += new_start_offset + len(match.group(2))

            if start < len(ch.text):
                cur_cell.contents.append(ch[start:])

        elif ch.name.lower() == 'multicolumn':
            cur_cell.contents.append(ch)
            cur_cell.colspan = int(str(ch.args[0]))

        elif ch.name.lower() == 'multirow':
            cur_cell.contents.append(ch)
            cur_cell.rowspan = int(str(ch.args[0]))

        else:
            cur_cell.contents.append(ch)

    if cur_cell.contents:
        result.append(cur_cell)

    return result


def structurize_tabular_contents(root):
    result = []
    last_line = []
    for ch in root.contents:
        if isinstance(ch, TexSoup.TexNode) and ch.name.lower() in ('multicolumn', 'multirow', 'makecell'):
            last_line.append(ch)
        elif isinstance(ch, (TexSoup.TokenWithPosition, TexSoup.TexNode)):
            for tok in get_all_tokens(ch, include_command_names=False):
                tok = tok.strip()
                start = 0
                while start < len(tok.text):
                    new_start_offset = 0
                    match = NEWLINE_RE.search(tok.text[start:])
                    if match:
                        new_start_offset = match.start(2)
                        token_to_append = tok[start:start+new_start_offset]
                        if len(token_to_append.strip().text) > 0:
                            last_line.append(token_to_append)
                        result.append(last_line)
                        last_line = []
                    else:
                        break
                    start += new_start_offset + len(match.group(2))

                if start < len(tok.text):
                    last_line.append(tok[start:])
        else:
            print('achtung! {} {}'.format(type(ch), ch))
    if last_line:
        result.append(last_line)
    return Table([split_row_into_cells(row) for row in result])


def get_table_info(soup, extract_cells=True):
    for table in list(soup.find_all('table')):
        tabular = table.tabular or table.array
        found_some_content = False
        if tabular is None:
            for elem in table.contents:
                if isinstance(elem, TexSoup.TokenWithPosition) and elem.strip().startswith('$'):
                    found_some_content = True
                    yield (1,
                           True,
                           [soup.char_pos_to_line(tok.position)
                            for tok in get_all_tokens(elem)],
                           [tok.text for tok in get_all_tokens(elem, False)],
                           [])
        else:
            found_some_content = True
            yield (1,
                   True,
                   [soup.char_pos_to_line(tok.position)
                    for tok in get_all_tokens(tabular)],
                   [tok.text for tok in get_all_tokens(tabular, False)],
                   [])

        if found_some_content:
            caption = table.caption
            if not caption is None:
                yield (0,
                       True,
                       [soup.char_pos_to_line(tok.position)
                        for tok in get_all_tokens(caption)],
                       [tok.text for tok in get_all_tokens(caption, False)],
                       [])

            if extract_cells:
                try:
                    parsed_table = structurize_tabular_contents(tabular)

                    cell_positions = []
                    cell_tokens = []
                    row_groups = [(3, []) for _ in parsed_table.rows]
                    col_groups = [(4, []) for _ in parsed_table.rows[-1]]
                    cur_cell_i = 0
                    for row_i in range(len(parsed_table.rows)-1, -1, -1):
                        row = parsed_table.rows[row_i]
                        row_col_ids = [0]
                        for cell in row[:-1]:
                            row_col_ids.append(row_col_ids[-1] + cell.colspan)
                        for cell, col_i in zip(row[::-1], row_col_ids[::-1]):
                            cell_positions.append([soup.char_pos_to_line(tok.position) for tok in get_all_tokens(cell)])
                            cell_tokens.append(list(get_all_tokens(cell, False)))

                            if cell.colspan > 1:
                                cur_cols = { col_i + off for off in range(cell.colspan) }
                                new_col_group = list(set(cell
                                                         for col in cur_cols
                                                         for cell in col_groups[col][1]))
                                new_col_group.append(cur_cell_i)
                                cur_cols.add(len(col_groups))
                                col_groups.append((4, new_col_group))
                            else:
                                col_groups[col_i][1].append(cur_cell_i)
                            col_i -= cell.colspan

                            if cell.rowspan > 1:
                                cur_rows = { row_i + off for off in range(cell.rowspan) }
                                new_row_group = list(set(cell
                                                         for row in cur_rows
                                                         for cell in row_groups[row][1]))
                                new_row_group.append(cur_cell_i)
                                cur_rows.add(len(row_groups))
                                row_groups.append((3, new_row_group))
                            else:
                                row_groups[row_i][1].append(cur_cell_i)

                            cur_cell_i += 1

                    yield (2,
                           False,
                           cell_positions,
                           cell_tokens,
                           row_groups + col_groups
    #                        col_groups
                          )
                except:
                    print(traceback.format_exc())


def get_cell_text(c):
    return ''.join(t.text.strip('$')
                   for el in c.contents
                   for t in get_all_tokens(el, include_command_names=False))


def assign_multiple(costs, allows_multiple, max_iter=10, max_cost=numpy.inf):
    result = collections.defaultdict(set)
    unassigned_rows = numpy.arange(0, costs.shape[0], 1).astype('int')
    unassigned_cols = numpy.arange(0, costs.shape[1], 1).astype('int')
    for iter_i in range(max_iter):
        if unassigned_rows.shape[0] == 0 or unassigned_cols.shape[0] == 0:
            break
        row_idx_rel, col_idx_rel = scipy.optimize.linear_sum_assignment(costs[unassigned_rows, :][:, unassigned_cols])
        row_idx_abs = unassigned_rows[row_idx_rel]
        col_idx_abs = unassigned_cols[col_idx_rel]

        assigned_rows = set()
        assigned_cols = set()
        for row_i, col_i in zip(row_idx_abs, col_idx_abs):
            if costs[row_i, col_i] < max_cost:
                assigned_rows.add(row_i)
                assigned_cols.add(col_i)
                result[row_i].add(col_i)
                print('assigned', iter_i, row_i, col_i, costs[row_i, col_i])

        if not assigned_cols:
            break
        single_assignment_constraint = set()
        for row_i in assigned_rows:
            if len(result[row_i]) > 0 and not allows_multiple[row_i]:
                single_assignment_constraint.add(row_i)
        unassigned_rows = numpy.setdiff1d(unassigned_rows, list(single_assignment_constraint))
        unassigned_cols = numpy.setdiff1d(unassigned_cols, list(assigned_cols))
    return result


NGRAM_SIM_WEIGHTS = (
    (5, 0.4),
    (4, 0.3),
    (3, 0.2),
    (2, 0.05),
    (1, 0.05)
)
FUZZY_DIST_PENALTY = 0.3
def ngram_dist(cell_txt, box_txt, empty_res=100):
    if not (cell_txt and box_txt):
        return empty_res
    if cell_txt == box_txt:
        return 0
    result = 0
    for ngram_len, weight in NGRAM_SIM_WEIGHTS:
        result += weight * ngram.NGram.compare(box_txt, box_txt, N=ngram_len)
    return 1.0 - result + FUZZY_DIST_PENALTY


FORCE_LINE_BREAK = r'\\'
def multiline_ngram_dist(cell_txt, box_txt, empty_res=100):
    if FORCE_LINE_BREAK in cell_txt:
        lines = cell_txt.split(FORCE_LINE_BREAK)
        return min(ngram_dist(line.strip(), box_txt, empty_res=empty_res)
                   for line in lines
                   if line.strip())
    else:
        return ngram_dist(cell_txt, box_txt, empty_res=empty_res)


def strict_cell_dist(cell_txt, box_txt, empty_res=100):
    if not (cell_txt and box_txt):
        return empty_res
    cell_lines = cell_txt.split(FORCE_LINE_BREAK)
    res = 0 if any(box_txt == l for l in cell_lines) else 1
#     print(res, '!!!!!!', cell_txt, box_txt)
    return res


def reassign_boxes(pdf, page_i, found_boxes, cells,
                   max_assign_dist=0.90, pos_factor=0.0,
                   max_em_iters=10, pos_diff_eps=1e-3,
                   max_assign_iter=10,
                   text_dist=multiline_ngram_dist):
    found_boxes = list(set(map(tuple, found_boxes)))
    found_boxes.sort(key=lambda b: (b[0], b[1]))

    cropbox = pdf.get_page(page_i-1)[1].cropbox
    box_texts = [pdf.get_text(page_i-1, [convert_coords_to_pq(fb, cropbox)])
                 for fb in found_boxes]
#     print('box_texts', list(zip(range(len(box_texts)), box_texts)))
    box_centers = numpy.array([get_box_center(fb) for fb in found_boxes])
    box_centers -= box_centers.min(0)
    box_centers /= box_centers.max(0)
    mean_box_center = box_centers.mean(0)
    max_center_dist = numpy.nan_to_num(scipy.spatial.distance.pdist(box_centers)).max()

    cell_texts = [''.join(t.text for t in c) for c in cells]
#     print('cell_texts', list(zip(range(len(cell_texts)), cell_texts)))
    cell_allows_multiple_boxes = [r'\\' in t for t in cell_texts]

    text_dist = numpy.array([[text_dist(ct, bt) for bt in box_texts]
                             for ct in cell_texts])

    cur_mapping = assign_multiple(text_dist,
                                  cell_allows_multiple_boxes,
                                  max_iter=max_assign_iter,
                                  max_cost=max_assign_dist)
    for iter_i in range(max_em_iters):
        cur_centers_dist = numpy.array([[min((scipy.spatial.distance.euclidean(box_center, cell_box)
                                              for cell_box in cur_mapping[cell_i]),
                                             default=max_center_dist)
                                         for box_center in box_centers]
                                        for cell_i in range(len(cell_texts))])
        cur_dist = (1 - pos_factor) * text_dist + pos_factor * cur_centers_dist
        new_mapping = assign_multiple(cur_dist,
                                      cell_allows_multiple_boxes,
                                      max_iter=max_assign_iter,
                                      max_cost=max_assign_dist)
        if new_mapping == cur_mapping:
            break
        cur_mapping = new_mapping

    result = [[found_boxes[box_i] for box_i in cur_mapping.get(cell_i, [])]
              for cell_i in range(len(cells))]
    return result


def common_substring_len(a, b):
    matcher = difflib.SequenceMatcher(r'\\'.__contains__, a=a, b=b)
    return matcher.find_longest_match(0, len(a), 0, len(b)).size


def reassign_boxes_greedy(pdf, page_i, found_boxes, cells, max_iters=3, ignore_out_of_cb=True):
    found_boxes = list(set(map(tuple, found_boxes)))
    found_boxes.sort(key=lambda b: (b[0], b[1]))

    cropbox = pdf.get_page(page_i-1)[1].cropbox
    cbx1, cby1, cbx2, cby2 = cropbox
    box_texts = [pdf.get_text(page_i-1, [convert_coords_to_pq(fb, cropbox) + PQ_BBOX_PAD])
                 for fb in found_boxes]
#     print('box_texts', list(zip(range(len(box_texts)), box_texts)))
    box_centers = numpy.array([get_box_center(fb) for fb in found_boxes])
#     print(cropbox)
#     print([convert_coords_to_pq(fb, cropbox) for fb in found_boxes])

    cell_texts = [''.join(t.text for t in c) for c in cells]
#     print('cell_texts', list(zip(range(len(cell_texts)), cell_texts)))

    cell2boxes = [set() for _ in cells]
    similarities = [(cell_i, box_i, sim)
                    for cell_i, cell_txt in enumerate(cell_texts)
                    for box_i, box_txt in enumerate(box_texts)
#                     for bx1, by1, bx2, by2 in [convert_coords_to_pq(found_boxes[box_i], cropbox)]
#                     if by2 <= cby2 and bx2 <= cbx2 # manage only fully visible boxes
                    for sim in [common_substring_len(cell_txt, box_txt)]
                    if sim > 1]
    similarities.sort(key=lambda p: p[-1], reverse=True)
#     print(similarities)

    b_idx = collections.defaultdict(set)
    for i, (cell_i, box_i, _) in enumerate(similarities):
        b_idx[box_i].add(i)

    assigned_boxes = set()
    for iter_i in range(3):
        if len(assigned_boxes) == len(found_boxes):
            break

        last_iter = iter_i == (max_iters - 1)

        for i, (cell_i, box_i, sim) in enumerate(similarities):
            if box_i in assigned_boxes:
                continue
            cell_boxes = cell2boxes[cell_i]
            if len(cell_boxes) == 0:
                cell_boxes.add(box_i)
                assigned_boxes.add(box_i)
#                 print('first', cell_i, cell_texts[cell_i], box_i, box_texts[box_i], sim)
            else:
                box_dists = { other_cell_i : min((scipy.spatial.distance.euclidean(box_centers[box_i], box_centers[cell_box_i])
                                                  for cell_box_i in cell2boxes[other_cell_i]),
                                                 default=numpy.inf)
                             for other_pos_i in b_idx[box_i]
                             for other_cell_i in [similarities[other_pos_i][0]]
                             if len(cell2boxes[other_cell_i]) > 0 }
                sorted_dists = sorted(box_dists.items(), key=lambda p: p[1])
                cur_cell_dist = box_dists[cell_i]
                closest_other_cell_i, closest_other_dist = sorted_dists[0]
                rel_sim = sim / len(box_texts[box_i])
                if (closest_other_cell_i == cell_i and len(box_dists) > 1 and rel_sim > 0.99) or last_iter:
#                     print('add', cell_i, cell_texts[cell_i], box_i, box_texts[box_i], sim, rel_sim)
                    cell_boxes.add(box_i)
                    assigned_boxes.add(box_i)
#                 else:
#                     print('skip', cell_i, cell_texts[cell_i], box_i, box_texts[box_i], sim, rel_sim)

    result = [[found_boxes[box_i] for box_i in cell_box_ids]
              for cell_box_ids in cell2boxes]
    return result
