import os, glob, json, tqdm, pandas, pickle, rtree, gc, toposort, joblib, numpy, math

from torch.utils.data import Dataset

from imgaug import augmenters as iaa
from imgaug import imgaug as ia
from PIL import Image

from IPython.display import display

from prepare_images_utils import *
from latex_dataset import *


IN_IMG_SUFFIX = '_in.png'
OUT_JSON_SUFFIX = '_out.json'


def load_image_with_boxes(img_id, mode='L'):
    img = load_image_opaque(img_id + IN_IMG_SUFFIX, mode=mode)
    with open(img_id + OUT_JSON_SUFFIX, 'r') as f:
        boxes = json.load(f)
    return img, boxes


def prepare_img_boxes_for_nn_crop(img, boxes, shape=(800, 400)):
    cats, just_boxes = zip(*boxes)
    cats = numpy.array(cats)
    just_boxes = numpy.array(just_boxes) * POINTS_TO_PIXELS_FACTOR
    just_boxes = just_boxes[:, [1, 0, 3, 2]] # x1, y1, x2, y2
    cropbox = numpy.array((just_boxes[:, 0].min(),
                           just_boxes[:, 1].min(),
                           just_boxes[:, 2].max(),
                           just_boxes[:, 3].max())).astype('int')

    res_in_img = Image.new('L', shape, 255)
    res_in_img.paste(img.crop(cropbox))

    just_boxes -= cropbox[[0, 1, 0, 1]]
    just_boxes = numpy.clip(just_boxes,
                            (0, 0, 0, 0),
                            (shape[0], shape[1], shape[0], shape[1]))
    boxes_area = (just_boxes[:, 2] - just_boxes[:, 0]) * (just_boxes[:, 3] - just_boxes[:, 1])
    good_boxes = numpy.where(boxes_area > 0)[0]
    return (numpy.array(res_in_img).astype('float32'),
            cats[good_boxes],
            just_boxes[good_boxes])


def prepare_img_boxes_for_nn_scale(img, boxes, shape=(850, 1100)):
    cats, just_boxes = zip(*boxes)
    cats = numpy.array(cats)
    just_boxes = numpy.array(just_boxes) * POINTS_TO_PIXELS_FACTOR
    just_boxes = just_boxes[:, [1, 0, 3, 2]] # x1, y1, x2, y2

    scale_ratio = min(shape[0] / img.size[0], shape[1] / img.size[1])
    res_in_img = img.resize((int(img.size[0] * scale_ratio),
                             int(img.size[1] * scale_ratio)))
    if res_in_img.size[0] < shape[0] or res_in_img.size[1] < shape[1]:
        buf = Image.new('L', shape, 255)
        buf.paste(res_in_img)
        res_in_img = buf

    return (numpy.array(res_in_img).astype('float32'),
            cats,
            just_boxes * scale_ratio)


def filter_by_intersection(big_box, boxes_to_filter, threshold=0.97):
    return [i for i, b in enumerate(boxes_to_filter)
            if box_inter_area(big_box, b) / box_area(b) >= threshold]


def group_by_intersection(criterion_boxes, boxes_to_group, threshold=0.97):
    result = collections.defaultdict(set)
    for group_i, big_box in enumerate(criterion_boxes):
        box_idx = filter_by_intersection(big_box, boxes_to_group)
        for box_i in box_idx:
            result[box_i].add(group_i)
    return result


def get_biggest_box(boxes):
    return boxes[numpy.argmax(list(map(box_area, boxes)))]


def make_grid(boxes):
    if len(boxes) == 0:
        return []

    rindex = rtree.index.Index(interleaved=True)
    for box_i, box in enumerate(boxes):
        rindex.insert(box_i, box)

    y1s, x1s, y2s, x2s = zip(*boxes)
    min_y, min_x, max_y, max_x = min(y1s), min(x1s), max(y2s), max(x2s)

    neighborhood = []
    for box_i, box in enumerate(boxes):
        box_y1, box_x1, box_y2, box_x2 = box

        immediate_left_neigh_border = max((boxes[i][1] for i in
                                           rindex.intersection((box_y1, min_x-1e-3, box_y2, box_x1-1e-3))),
                                          default=min_x)
        immediate_right_neigh_border = min((boxes[i][3] for i in
                                            rindex.intersection((box_y1, box_x2+1e-3, box_y2, max_x+1e-3))),
                                           default=max_x)
        immediate_upper_neigh_border = max((boxes[i][0] for i in
                                            rindex.intersection((min_y-1e-3, box_x1, box_y1-1e-3, box_x2))),
                                           default=min_y)
        immediate_lower_neigh_border = min((boxes[i][2] for i in
                                            rindex.intersection((box_y2+1e-3, box_x1, max_y+1e-3, box_x2))),
                                           default=max_y)

        cur_box_neighborhood = {}

        cur_box_neighborhood['lower'] = list(rindex.intersection((box_y2+1e-3,
                                                                  immediate_left_neigh_border,
                                                                  immediate_lower_neigh_border+1e-3,
                                                                  immediate_right_neigh_border)))

        cur_box_neighborhood['right'] = list(rindex.intersection((immediate_upper_neigh_border,
                                                                  box_x2+1e-3,
                                                                  immediate_lower_neigh_border,
                                                                  immediate_right_neigh_border+1e-3)))

        neighborhood.append(cur_box_neighborhood)
    return neighborhood


def get_axis_inter(a, b, axis):
    return max(a[axis], b[axis]), min(a[axis+2], b[axis+2])


def get_interbox_space_overlapping(a, b):
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, iy2 = get_axis_inter(a, b, 0)
    ix1, ix2 = get_axis_inter(a, b, 1)
    if iy1 <= iy2: # intersect vertically
        if ax2 < bx1: # a is to the left of b
            return (iy1, ax2, iy2, bx1)
        else: # b is to the right a
            return (iy1, bx2, iy2, ax1)
    elif ix1 <= ix2: # intersect horizontally
        if ay2 < by1: # a is upper than b
            return (ay2, ix1, by1, ix2)
        else:
            return (by2, ix1, ay1, ix2)
    else:
        print(a, b)
        print(iy1, iy2)
        print(ix1, ix2)
        raise Exception('WAT?!?!? we must not have got here!')


def just_box_union(boxes):
    y1s, x1s, y2s, x2s = zip(*boxes)
    return [min(y1s), min(x1s), max(y2s), max(x2s)]


def get_interbox_space_fuzzy(a, b):
    return just_box_union((a, b))


def make_bi_slope(width):
    even = width % 2 == 0
    slope = numpy.arange(1, 0, -2 / width)[::-1][-width//2:]
    return numpy.concatenate([(slope if even else slope[:-1]), slope[::-1]])


def make_diagonal_mask(box, is_left):
    y1, x1, y2, x2 = box
    width = x2 - x1
    height = y2 - y1
    if height > width:
        width, height = height, width
        transposed = True
    else:
        transposed = False
    x_steps = numpy.round(width / float(height) * numpy.arange(0, height, 1)).astype('int')
    line_filler = numpy.arange(1, 0, -1 / width)[:width]
    rows = ([line_filler] + 
        [numpy.concatenate([line_filler[1:x+1][::-1], line_filler[:-x]])
         for x in x_steps[1:]])
    result = numpy.array(rows)
    if transposed:
        result = result.T
    if is_left:
        result = result[:, ::-1]
    return result


def get_interbox_mask_direction(a, b):
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    return (ay1 - by1) * (ax1 - bx1) < 0


def make_fill_const(box, value=1):
    return value


def bbox_center_cv2(box):
    y1, x1, y2, x2 = box
    return (int((x2 + x1) / 2), int((y2 + y1) / 2))


MIN_THICKNESS = 6
MAX_THICKNESS = 40
def calc_intercell_line_mask_params(cell1, cell2, direction):
    center1 = bbox_center_cv2(cell1)
    center2 = bbox_center_cv2(cell2)
    if direction in ('same_row', 'right'):
        thickness = min(cell1[2] - cell1[0],
                        cell2[2] - cell2[0])
    elif direction in ('same_col', 'lower'):
        thickness = min(cell1[3] - cell1[1],
                        cell2[3] - cell2[1])
    else:
        raise Exception('We must not get here! {}'.format(direction))
    thickness = int(max(MIN_THICKNESS,
                        min(thickness * 0.7,
                            scipy.spatial.distance.euclidean(center1, center2),
                            MAX_THICKNESS)))
    return center1, center2, thickness


def get_intercell_line_bbox(cell1, cell2, direction):
#     (c1x, c1y), (c2x, c2y), thickness = calc_intercell_line_mask_params(cell1, cell2, direction)
#     return min(c1y, c2y), min(c1x, c2x), max(c1y, c2y), max(c1x, c2x)
    return just_box_union([cell1, cell2])


def draw_intercell_mask(output, cell1, cell2, direction, value=1.0):
    center1, center2, thickness = calc_intercell_line_mask_params(cell1, cell2, direction)
    cv2.line(output, center1, center2, (value,),
             thickness=thickness, lineType=cv2.LINE_AA)


def make_mask_for_nn_base(size, total_classes, boxes_by_channel):
    result = numpy.zeros((total_classes, ) + size, dtype='float32')
    for channel, boxes in enumerate(boxes_by_channel):
        for box in boxes:
            y1, x1, y2, x2 = (int(math.floor(box[0])),
                              int(math.floor(box[1])),
                              int(math.ceil(box[2])),
                              int(math.ceil(box[3])))
            result[channel, y1:y2, x1:x2] = 1
    return result

DET_MASK_CHANNELS = ['caption', 'body']
DET_CAPTION_CHANNEL_I = 0
DET_BODY_CHANNEL_I = 1
TOTAL_DET_CLASSES = len(DET_MASK_CHANNELS)
def make_mask_for_nn_det(size, box_cats, boxes_on_image):
    boxes_by_cat = collections.defaultdict(list)
    for cat, bbox in zip(box_cats, boxes_on_image.bounding_boxes):
        boxes_by_cat[cat].append((bbox.y1, bbox.x1, bbox.y2, bbox.x2))
    captions = boxes_by_cat[0]
    bodies = boxes_by_cat[1]
    return make_mask_for_nn_base(size, TOTAL_DET_CLASSES, [captions, bodies])


INT_MASK_CHANNELS = ['body', 'cell', 'same_row_other_col', 'same_col_other_row']
TOTAL_INT_CLASSES = len(INT_MASK_CHANNELS)
INTERCELL_LINE_WIDTH = 10
def make_mask_for_nn_intracell(size, box_cats, boxes_on_image):
    result = numpy.zeros((TOTAL_INT_CLASSES, ) + size, dtype='float32')

    boxes_by_cat = collections.defaultdict(list)
    for cat, bbox in zip(box_cats, boxes_on_image.bounding_boxes):
        boxes_by_cat[cat].append((bbox.y1, bbox.x1, bbox.y2, bbox.x2))

    if len(boxes_by_cat[1]) == 0:
        return result

    body = get_biggest_box(boxes_by_cat[1])
    cells = boxes_by_cat[2]
    rows = boxes_by_cat[3]
    cols = boxes_by_cat[4]

    cell2rows = group_by_intersection(rows, cells)
    cell2cols = group_by_intersection(cols, cells)

    boxes_by_channel = [[body], cells]
    result = make_mask_for_nn_base(size, TOTAL_INT_CLASSES, boxes_by_channel)

    grid = make_grid(cells)
    for i, cur_box in enumerate(cells):
        cur_rows = cell2rows[i]
        cur_cols = cell2cols[i]
        for direction, neigh_boxes_idx in grid[i].items():
            for neigh_box_i in neigh_boxes_idx:
                neigh_box = cells[neigh_box_i]
                # here we assume that we are going to the right and bottom
                same_rows = cur_rows <= cell2rows[neigh_box_i]
                same_cols = cur_cols <= cell2cols[neigh_box_i]
                if same_rows and same_cols:
#                     print('same_rows and same_cols')
#                     print(i, neigh_box_i)
#                     print(cur_box, cells[neigh_box_i])
#                     print(cur_rows, cur_cols)
#                     print(cell2rows[neigh_box_i], cell2cols[neigh_box_i])
#                     continue
                    raise Exception('Different cells but same row and col?!?!?!?!?!')
                if not (same_rows or same_cols):
                    continue
                channel = 2 if same_rows else 3
                draw_intercell_mask(result[channel],
                                    cur_box,
                                    neigh_box,
                                    'same_row' if same_rows else 'same_col',
                                    1)

    return result


MASK_COLORS = numpy.array([
    [255,   0,   0],
    [  0, 255,   0],
    [  0,   0, 255],
    [255, 255,   0],
    [255,   0, 255],
])
def mask_to_img(mask, color_offset=0, fixed_norm=True):
    norm = float(mask.shape[0]) if fixed_norm else numpy.expand_dims(mask.sum(0), -1)
    colored = numpy.tensordot(mask, MASK_COLORS[color_offset:color_offset+mask.shape[0]], (0, 0))
    avg = numpy.nan_to_num(colored / norm)
    return avg / 255.0


def calc_loss_weights(mask, channels=None, edge_add_weight=10.0, laplacian_ksize=9, edge_th=1.1):
    result = numpy.ones_like(mask)
    if channels is None:
        channels = list(range(mask.shape[1]))
    for sample_i in range(mask.shape[0]):
        for channel_i in channels:
            edges = numpy.absolute(cv2.Laplacian(mask[sample_i, channel_i],
                                                 cv2.CV_32F,
                                                 ksize=laplacian_ksize))
            edges = numpy.where(edges > edge_th, 1, 0)
            if edges.max() > 0:
                result[sample_i, channel_i] += edge_add_weight * edges
    return result


def my_augment_bounding_boxes(augmenter, bounding_boxes_on_images, boxes_cats_by_image):
    kps_ois = []
    for bbs_oi in bounding_boxes_on_images:
        kps = []
        for bb in bbs_oi.bounding_boxes:
            kps.extend(bb.to_keypoints())
        kps_ois.append(ia.KeypointsOnImage(kps, shape=bbs_oi.shape))

    kps_ois_aug = augmenter.augment_keypoints(kps_ois)

    result = []
    for img_idx, kps_oi_aug in enumerate(kps_ois_aug):
        img_cats = boxes_cats_by_image[img_idx]
        bbs_aug = []
        aug_cats = []
        for i in range(len(kps_oi_aug.keypoints) // 4):
            bb_kps = kps_oi_aug.keypoints[i*4:i*4+4]
            x1 = numpy.clip(min([kp.x for kp in bb_kps]), 0, None)
            x2 = numpy.clip(max([kp.x for kp in bb_kps]), 0, None)
            y1 = numpy.clip(min([kp.y for kp in bb_kps]), 0, None)
            y2 = numpy.clip(max([kp.y for kp in bb_kps]), 0, None)
            area = (x2 - x1) * (y2 - y1)
            if area > 0 and x1 >= 0 and y1 >= 0:
                bbs_aug.append(
                    bounding_boxes_on_images[img_idx].bounding_boxes[i].copy(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2
                    )
                )
                aug_cats.append(img_cats[i])
        result.append((ia.BoundingBoxesOnImage(bbs_aug,
                                               shape=kps_oi_aug.shape),
                       aug_cats))
    return result


def prepare_det_batch(batch_image_ids, augmenter, out_shape=(800, 1024)):
    images, box_cats, boxes = zip(*[prepare_img_boxes_for_nn_scale(*load_image_with_boxes(img_id),
                                                                   shape=out_shape)
                                    for img_id in batch_image_ids])

    det = augmenter.to_deterministic() if not augmenter.deterministic else augseq

    images_aug = det.augment_images(numpy.array(images).astype('uint8')).astype('float32') / 255

    boxes = [ia.BoundingBoxesOnImage([ia.BoundingBox(*box)
                                      for box in img_boxes],
                                     img.shape)
             for img, img_boxes in zip(images, boxes)]
    boxes_aug_with_cats = my_augment_bounding_boxes(det, boxes, box_cats)

    mask = numpy.array([make_mask_for_nn_det(img.shape, img_box_cats, img_boxes)
                        for img_name, img, (img_boxes, img_box_cats)
                        in zip(batch_image_ids, images_aug, boxes_aug_with_cats)])

    boxes_aug_lists_with_cats = []
    for img_boxes, img_box_cats in boxes_aug_with_cats:
        img_boxes_with_cats = collections.defaultdict(list)
        for b, c in zip(img_boxes.bounding_boxes, img_box_cats):
            img_boxes_with_cats[c].append((b.y1, b.x1, b.y2, b.x2)) # we throw away 0 category
        boxes_aug_lists_with_cats.append(img_boxes_with_cats)

    return (batch_image_ids,
            numpy.expand_dims(numpy.array(images_aug), 1),
            mask,
            calc_loss_weights(mask, channels=[]),
            boxes_aug_lists_with_cats)


def prepare_int_batch(batch_image_ids, augmenter, out_shape=(800, 400)):
    images, box_cats, boxes = zip(*[prepare_img_boxes_for_nn_crop(*load_image_with_boxes(img_id),
                                                                  shape=out_shape)
                                    for img_id in batch_image_ids])

    det = augmenter.to_deterministic() if not augmenter.deterministic else augseq

    images_aug = det.augment_images(numpy.array(images).astype('uint8')).astype('float32') / 255

    boxes = [ia.BoundingBoxesOnImage([ia.BoundingBox(*box)
                                      for box in img_boxes],
                                     img.shape)
             for img, img_boxes in zip(images, boxes)]
    boxes_aug_with_cats = my_augment_bounding_boxes(det, boxes, box_cats)

    mask = numpy.array([make_mask_for_nn_intracell(img.shape, img_box_cats, img_boxes)
                        for img_name, img, (img_boxes, img_box_cats)
                        in zip(batch_image_ids, images_aug, boxes_aug_with_cats)])

    boxes_aug_lists_with_cats = []
    for img_boxes, img_box_cats in boxes_aug_with_cats:
        img_boxes_with_cats = collections.defaultdict(list)
        for b, c in zip(img_boxes.bounding_boxes, img_box_cats):
            img_boxes_with_cats[c-1].append((b.y1, b.x1, b.y2, b.x2)) # we throw away 0 category
        boxes_aug_lists_with_cats.append(img_boxes_with_cats)

    return (batch_image_ids,
            numpy.expand_dims(numpy.array(images_aug), 1),
            mask,
            calc_loss_weights(mask, channels=[1]),
            boxes_aug_lists_with_cats)


def data_gen(image_ids, augmenter, batch_gen_func=prepare_int_batch, batch_size=32):
    while True:
        yield batch_gen_func(numpy.random.choice(image_ids, size=batch_size),
                             augmenter)


class SegmDataset(Dataset):
    def __init__(self, all_image_ids, augmenter, batch_gen_func):
        self.all_image_ids = all_image_ids
        self.augmenter = augmenter
        self.batch_gen_func = batch_gen_func

    def __len__(self):
        return len(self.all_image_ids)

    def __getitem__(self, i):
        (batch_image_ids,
         in_img,
         mask,
         loss_weights,
         boxes_aug) = self.batch_gen_func([self.all_image_ids[i]],
                                          self.augmenter)
        boxes_aug_str = pickle.dumps(boxes_aug[0])
        return (batch_image_ids,
                in_img[0],
                mask[0],
                loss_weights[0],
                boxes_aug_str)


def check_image_functor(img, batch_gen_func, augmenter):
    try:
        batch_gen_func([img], augmenter)
        return True, img
    except:
        print(traceback.format_exc())
        return False, img


def leave_only_valid_samples(image_ids, batch_gen_func, augmenter, n_jobs=-1):
    valid = []
    errors = []
    for is_valid, img in joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(check_image_functor)(img,
                                                                                            batch_gen_func,
                                                                                            augmenter)
                                                        for img in image_ids):
        if is_valid:
            valid.append(img)
        else:
            errors.append(img)
    return valid, errors


imgaug_pipeline = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.5),
#     iaa.Affine(rotate=iap.DiscreteUniform(0, 3) * 90, cval=(0, 0, 255)),
    iaa.GaussianBlur(sigma=(0.0, 2.0)),
    iaa.CropAndPad(px=20, pad_cval=255),
    iaa.CropAndPad(percent=(-0.20, 0.20), pad_cval=255),
])

fake_imgaug_pipeline = iaa.Sequential([])