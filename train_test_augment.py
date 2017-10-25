#!/usr/bin/env python

import os, random, glob, shutil, itertools
from PIL import Image
from joblib import Parallel, delayed

IN_SUFFIX = '_in.png'
OUT_SUFFIX = '_out.png'
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2

RESIZE_TO_X = 500
RESIZE_TO_Y_FACTOR = 1.32
RESIZE_TO = (RESIZE_TO_X, int(RESIZE_TO_X * RESIZE_TO_Y_FACTOR))
WINDOW_SIZE = RESIZE_TO
STRIDE = (50, 50)
OUT_SCALE_TO = (256, 256)
OUT_MODE = 'RGB'
ROTATIONS = (0, 90, 180, 270)
SCALES = [(1, 1), (1, 1.2), (1.2, 1)]


def augment_image_deterministic(fname, out_dir,
                                window_size=WINDOW_SIZE, stride=STRIDE, scales=SCALES,
                                out_scale_to=OUT_SCALE_TO, rotations=ROTATIONS):
    out_i = 0
    base_fname = os.path.splitext(os.path.basename(fname))[0]
    if '_' in base_fname:
        base_id, kind = base_fname.rsplit('_', 1)
    else:
        base_id = base_fname
        kind = None
    src_img = Image.open(fname).resize(RESIZE_TO) #.convert('RGB').convert(OUT_MODE)
    out_fname = os.path.join(out_dir,
                             '_'.join((base_id, str(out_i)) + (() if kind is None else (kind,))) + '.png')
    src_img.save(out_fname)
    return [(1, 1, 0, 0, 0, out_fname)]
    #alpha = src_img.convert('RGBA').split()[-1]
    #bg = Image.new("RGBA", src_img.size, (255, 255, 255, 255))
    #bg.paste(src_img, mask=alpha)
    #src_img = bg.convert(OUT_MODE)

    offset_gen = itertools.product(range(0,
                                         src_img.size[0] - window_size[0],
                                         stride[0]),
                                   range(0,
                                         src_img.size[1] - window_size[1],
                                         stride[1]))
    result = []
    for scale_x, scale_y in scales:
        scaled_image = src_img.resize((int(src_img.size[0] * scale_x),
                                       int(src_img.size[1] * scale_y)),
                                      resample=Image.BILINEAR)
        for x_off, y_off in offset_gen:
            new_image = scaled_image.crop((x_off,
                                           y_off,
                                           x_off + window_size[0],
                                           y_off + window_size[1]))
            new_image.thumbnail(out_scale_to)
            for angle in rotations:
                out_fname = os.path.join(out_dir,
                                         '_'.join((base_id, str(out_i)) + (() if kind is None else (kind,))) + '.png')
                new_image.rotate(angle).save(out_fname)
                out_i += 1
                result.append((scale_x, scale_y, x_off, y_off, angle, out_fname))
    return result


def copy_one_augmented(src_dir, prefix, target_dir):
    augment_image_deterministic(os.path.join(src_dir,
                                             prefix + IN_SUFFIX),
                                target_dir)
    augment_image_deterministic(os.path.join(src_dir,
                                             prefix + OUT_SUFFIX),
                                target_dir)


def copy_all_augmented(src_dir, prefixes, target_dir, n_jobs=-1):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    Parallel(n_jobs=n_jobs)(delayed(copy_one_augmented)(src_dir, prefix, target_dir)
                            for prefix in prefixes)


def copy_all_svg(svg_dir, prefixes, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for prefix in prefixes:
        src_file = os.path.join(svg_dir, prefix + '.svg')
        if os.path.exists(src_file):
            shutil.copy2(src_file,
                         os.path.join(target_dir, prefix + '.svg'))


def shuffle_split(in_dir='/notebook/data/4_inout_pairs/', out_dir='/notebook/data/5_ready/', svg_dir='/notebook/data/3_prepared_images/', train_size=TRAIN_SIZE, val_size=VAL_SIZE):
    all_sample_names = list(set(os.path.basename(fname)[:-len(IN_SUFFIX)]
                                for fname in glob.glob(os.path.join(in_dir, '*' + IN_SUFFIX))))
    random.shuffle(all_sample_names)

    total_samples = len(all_sample_names)
    train_number = int(total_samples * train_size)
    val_number = int(total_samples * val_size)

    train_prefixes = all_sample_names[:train_number]
    val_prefixes = all_sample_names[train_number:train_number+val_number]
    test_prefixes = all_sample_names[train_number+val_number:]

    copy_all_augmented(in_dir,
                       train_prefixes,
                       os.path.join(out_dir, 'train'))
    copy_all_augmented(in_dir,
                       val_prefixes,
                       os.path.join(out_dir, 'val'))
    copy_all_svg(svg_dir,
                 test_prefixes,
                 os.path.join(out_dir, 'test'))


if __name__ == '__main__':
    shuffle_split()
