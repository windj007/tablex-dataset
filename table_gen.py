from latex_dataset import *

CONTENT_TYPES = ['number', 'number_sized', 'label', 'long_text']
CONTENT_TYPES_P = [0.25, 0.25, 0.25, 0.25]
def gen_content_type():
    return numpy.random.choice(CONTENT_TYPES, p=CONTENT_TYPES_P)


HEADER_CHILDREN_N = [2, 3]
HEADER_CHILDREN_N_P = [0.8, 0.2]
def gen_header_children_n():
    return numpy.random.choice(HEADER_CHILDREN_N, p=HEADER_CHILDREN_N_P)



_USED_WORDS = set()
def reset_word_gen():
    _USED_WORDS = set()


WORD_LEN_MEAN = 4
WORD_LEN_STD = 3
ALLOWED_CHARS = list(''.join(map(chr,
                                 itertools.chain(range(ord('A'), ord('Z')),
                                                 range(ord('a'), ord('z'))))) + '-.,*')
def gen_word_impl(mean=WORD_LEN_MEAN, std=WORD_LEN_STD):
    word_len = max(2, int(numpy.round(numpy.random.normal(WORD_LEN_MEAN, WORD_LEN_STD))))
    return ''.join(numpy.random.choice(ALLOWED_CHARS, word_len))

BAD_WORD_RE = re.compile('fi|ff', re.I)
def gen_word(*args, **kwargs):
    while True:
        cand = gen_word_impl(*args, **kwargs)
        if not cand in _USED_WORDS and not BAD_WORD_RE.search(cand):
            _USED_WORDS.add(cand)
            return cand


LONG_TEXT_MAX_LINE = 3
FORCE_LINE_BREAK = r'\\'
def gen_long_text(words_n=None, max_len=3, line_len = LONG_TEXT_MAX_LINE, wrap_makecell=True):
    if words_n is None:
        words_n = numpy.random.randint(2, max_len)
    words = [el
             for i in range(words_n)
             for el in [gen_word(),] + ([FORCE_LINE_BREAK] if (i + 1) % line_len == 0 else [])
            ]
    if words[-1] == FORCE_LINE_BREAK:
        words = words[:-1]
    content = ' '.join(words)
    if FORCE_LINE_BREAK in content and wrap_makecell:
        return r' \makecell{{ {} }} '.format(content)
    else:
        return content


TITLE_WORDS_N = [1, 2, 3, 4, 5, 6]
TITLE_WORDS_N_P = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]
assert sum(TITLE_WORDS_N_P) == 1
TITLE_LINE_MIN = 2
TITLE_LINE_MAX = max(TITLE_WORDS_N)
def gen_title(wrap_makecell=True):
    words_n = max(1, numpy.random.choice(TITLE_WORDS_N, p=TITLE_WORDS_N_P))
    line_size = numpy.random.randint(TITLE_LINE_MIN, max(TITLE_LINE_MIN, words_n) + 1)
    return gen_long_text(words_n=words_n, line_len=line_size, wrap_makecell=wrap_makecell)


ALIGN = list('clr')
ALIGN_P = [0.6, 0.2, 0.2]
def gen_align():
    return numpy.random.choice(ALIGN, p=ALIGN_P)


BORDERS = ['|', ' ']
BORDERS_P = [0.4, 0.6]
def gen_borders():
    return numpy.random.choice(BORDERS, p=BORDERS_P)


H_HEADER_TYPES = ['atomic', 'composite']
H_HEADER_TYPES_P = [0.8, 0.2]
V_HEADER_TYPES = ['atomic', 'composite']
V_HEADER_TYPES_P = [0.5, 0.5]
MAX_HEADER_DEPTH = 2
Header = collections.namedtuple('Header',
                                'type title content_type children format'.split(' '))
def gen_header(kind, depth=0):
    if depth < MAX_HEADER_DEPTH:
        if kind == 'h':
            header_type = numpy.random.choice(H_HEADER_TYPES, p=H_HEADER_TYPES_P)
        else:
            header_type = numpy.random.choice(V_HEADER_TYPES, p=V_HEADER_TYPES_P)
    else:
        header_type = 'atomic'
    title = gen_title()
    formatting = dict(align=gen_align(),
                      borders=gen_borders())
    if header_type == 'atomic':
        ct = gen_content_type()
        if ct == 'long_text':
            formatting['align'] = 'c'
        return Header(header_type, title, ct, [], formatting)
    elif header_type == 'composite':
        formatting['align'] = 'c'
        children_n = gen_header_children_n()
        children = [gen_header(kind, depth=depth+1) for _ in range(children_n)]
        return Header(header_type, title, '', children, formatting)
    raise NotImplemented()


def gen_headers(n, kind):
    return [gen_header(kind) for _ in range(n)]


def get_headers_depth(headers):
    return max((1 if h.type == 'atomic' else (get_headers_depth(h.children) + 1))
               for h in headers)


def iter_over_atomic_headers(headers):
    for h in headers:
        if h.type == 'atomic':
            yield h
        else:
            for h in iter_over_atomic_headers(h.children):
                yield h


def iter_over_headers_on_depth(headers, target_depth):
    if target_depth < 0:
        return
    if target_depth == 0:
        for h in headers:
            yield h
    else:
        for h in headers:
            for sub_h in iter_over_headers_on_depth(h.children, target_depth-1):
                yield sub_h


def iter_over_headers_all_depths(headers, max_depth):
    for h in headers:
        if h.type == 'atomic':
            yield [h] + [None for _ in range(max_depth - 1)]
        else:
            first = True
            for ch_row in iter_over_headers_all_depths(h.children, max_depth - 1):
                if first:
                    yield [h] + ch_row
                    first = False
                else:
                    yield [None] + ch_row

                    
def get_header_rows_for_col_headers(headers):
    raw_rows = iter_over_headers_all_depths(headers, get_headers_depth(headers))
    raw_rows = zip(*raw_rows)
    result = []
    for raw_row in raw_rows:
        row_out = []
        i = 0
        while i < len(raw_row):
            cur_h = raw_row[i]
            if cur_h is None:
                row_out.append(cur_h)
                i += 1
            elif cur_h.type == 'atomic':
                row_out.append(cur_h)
                i += 1
            else:
                row_out.append(cur_h)
                i += len(list(iter_over_atomic_headers(cur_h.children)))
        result.append(row_out)
    return result


def gen_number():
    return '{:.2f}'.format(numpy.random.rand() * 100)


def gen_number_sized():
    return ' '.join((gen_number(),
                     gen_word(mean=3, std=1)))


def gen_cell(cell_header, row_header):
    content_type = cell_header.content_type
    if content_type == 'number':
        return gen_number()
    elif content_type == 'number_sized':
        return gen_number_sized()
    elif content_type == 'label':
        return gen_word()
    elif content_type == 'long_text':
        return gen_long_text()
    raise NotImplemented()


def gen_hspacing():
    factor = numpy.random.beta(1.7, 10) * 15
    return r'\setlength{{\tabcolsep}}{{{:.2f}pt}}'.format(factor)


def gen_vspacing():
    factor = numpy.random.beta(1.5, 2) + 1
    return r'\renewcommand{{\arraystretch}}{{{:.2f}}}'.format(factor)


def gen_table_contents():
    reset_word_gen()
    col_headers = gen_headers(numpy.random.randint(2, 4), 'v')
    row_headers = gen_headers(numpy.random.randint(2, 10), 'h')
    cells = [[gen_cell(ch, rh) for ch in iter_over_atomic_headers(col_headers)]
             for rh in iter_over_atomic_headers(row_headers)]
    table_format = dict(attributes=[#r'\centering',
                                    gen_vspacing(),
                                    gen_hspacing()])
    return col_headers, row_headers, cells, table_format


MULTICOL_HEADERS_FORMAT = r'\multicolumn{{{children_n}}}{{{borders}{align}{borders}}}{{{title}}}'
def col_header_to_latex(header):
    if not header:
        return ''
    if header.type == 'atomic':
        return header.title
    format_args = dict(children_n=len(list(iter_over_atomic_headers(header.children))),
                       title=header.title)
    format_args.update(header.format)
    return MULTICOL_HEADERS_FORMAT.format(**format_args)


MULTIROW_HEADERS_FORMAT = r'\multirow{{{children_n}}}{{*}}{{{title}}}'
def row_header_to_latex(header):
    if not header:
        return ''
    if header.type == 'atomic':
        return header.title
    format_args = dict(children_n=len(list(iter_over_atomic_headers(header.children))),
                       title=header.title)
    format_args.update(header.format)
    return MULTIROW_HEADERS_FORMAT.format(**format_args)


def add_hline_if_needed(row_headers, row_length):
    if not row_headers:
        return ''
    first_border_i = -1
    for i, h in enumerate(row_headers):
        if not h is None and h.format['borders'] == '|':
            first_border_i = i
            break
    if first_border_i < 0:
        return ''
    elif first_border_i == 0:
        return r'\hline'
    else:
        return '\\cline{{{start}-{end}}}\n'.format(start=first_border_i+1,
                                                   end=row_length)


BASE_TABLE_TEMPLATE = r'''
\begin{{table}}
{table_format}
\begin{{tabular}}{{{tabular_format}}}
{tabular_contents}
\end{{tabular}}
\end{{table}}
'''
def table_contents_to_latex(col_headers, row_headers, cells_contents, table_format):
    col_headers_rows_n = get_headers_depth(col_headers)
    row_headers_cols_n = get_headers_depth(row_headers)
    rows = []
    for cur_col_headers_row in get_header_rows_for_col_headers(col_headers):
        row_cells = [''] * row_headers_cols_n
        row_cells.extend(map(col_header_to_latex, cur_col_headers_row))
        rows.append(row_cells)
    row_headers_iter = list(iter_over_headers_all_depths(row_headers, row_headers_cols_n))
    for cur_row_headers, line in zip(row_headers_iter, cells_contents):
        row_cells = [row_header_to_latex(h) for h in cur_row_headers] + line
        rows.append(row_cells)
    total_cols = len(rows[0])
    table_body = '\n'.join(add_hline_if_needed(cur_row_headers, total_cols) + ' ' + ' & '.join(cells) + r' \\'
                           for cur_row_headers, cells
                           in zip(([None] * col_headers_rows_n) + row_headers_iter, 
                                  rows))
    table_body += '\n' + add_hline_if_needed(row_headers_iter[-1], total_cols)

    borders = [h.format['borders'] for h in iter_over_atomic_headers(col_headers)]
    alignments = [h.format['align'] for h in iter_over_atomic_headers(col_headers)]
    tabular_format = ''.join(seq[i]
                             for i in range(len(borders))
                             for seq in (borders, alignments))
    tabular_format = borders[0] + ' '.join('c' * row_headers_cols_n) + tabular_format + borders[-1]

    return BASE_TABLE_TEMPLATE.format(caption='test table',
                                      table_format='\n'.join(table_format['attributes']),
                                      tabular_format=tabular_format,
                                      tabular_contents=table_body)


def check_generated_table(out_file):
    with open(out_file + '.tex', 'r') as f:
        tex = TexSoup.TexSoup(f.read())
    table = structurize_tabular_contents(tex.table.tabular)
    with open(out_file + '_0001_out.json', 'r') as f:
        boxes = json.load(f)
    found_cells_num = sum(1 for ch, _ in boxes if ch == 2)
    real_cells_num = sum(1
                         for row in table.rows
                         for cell in row
                         if ''.join(t.text for t in get_all_tokens(cell)))
    if real_cells_num != found_cells_num:
#         print('found', found_cells_num, 'real', real_cells_num)
        return False
    return True


def render_table(table_def, template_dir, out_file, print_latex_content=False, display_demo=False, on_wrong_parse='ignore'):
    with tempfile.TemporaryDirectory() as wd:
        for fname in os.listdir(template_dir):
            shutil.copy2(os.path.join(template_dir, fname), wd)

        our_latex_file = guess_main_latex_file(wd)
        with open(our_latex_file, 'r') as f:
            latex_template = f.read()
        table_contents = table_contents_to_latex(*table_def)
        latex_contents = latex_template.format(contents=table_contents)
        with open(our_latex_file, 'w') as f:
            f.write(latex_contents)
        if print_latex_content:
            print(latex_contents)
        compile_latex(wd)
        target_pdf_file = os.path.splitext(our_latex_file)[0] + '.pdf'
        target_pdf_filename = os.path.basename(target_pdf_file)

        pdf_latex_to_samples(os.path.basename(out_file),
                             wd,
                             our_latex_file,
                             target_pdf_file,
                             os.path.dirname(out_file),
                             get_table_info,
                             boxes_aggregator=aggregate_object_bboxes,
                             display_demo=display_demo)

        correct = check_generated_table(out_file)
        assert on_wrong_parse in ('delete', 'raise', 'ignore')
        if not correct:
            if on_wrong_parse == 'delete':
                for fname in glob.glob(out_file + '*'):
                    os.remove(fname)
            elif on_wrong_parse == 'raise':
                raise Exception('Wrong sample in {}'.format(out_file))
