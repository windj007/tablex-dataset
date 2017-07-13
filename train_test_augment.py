#!/usr/bin/env python

import os, random, glob, shutil, itertools
from PIL import Image


IN_SUFFIX = '_in.png'
OUT_SUFFIX = '_out.png'
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2

WINDOW_SIZE = (500, 500)
STRIDE = (50, 50)
OUT_SCALE_TO = (256, 256)
OUT_MODE = 'RGB'
ROTATIONS = (0, 90, 180, 270)
SCALES = [(1, 1), (1, 1.2), (1.2, 1)]

def augment_image_deterministic(fname, out_dir):
    out_i = 0
    base_fname = os.path.splitext(os.path.basename(fname))[0]
    base_id, kind = base_fname.rsplit('_', 1)
    src_img = Image.open(fname) #.convert('RGB').convert(OUT_MODE)
    #alpha = src_img.convert('RGBA').split()[-1]
    #bg = Image.new("RGBA", src_img.size, (255, 255, 255, 255))
    #bg.paste(src_img, mask=alpha)
    #src_img = bg.convert(OUT_MODE)

    offset_gen = itertools.product(range(0,
                                         src_img.size[0] - WINDOW_SIZE[0],
                                         STRIDE[0]),
                                   range(0,
                                         src_img.size[1] - WINDOW_SIZE[1],
                                         STRIDE[1]))
    for scale_x, scale_y in SCALES:
        scaled_image = src_img.resize((int(src_img.size[0] * scale_x),
                                       int(src_img.size[1] * scale_y)),
                                      resample=Image.BILINEAR)
        for x_off, y_off in offset_gen:
            new_image = scaled_image.crop((x_off,
                                           y_off,
                                           x_off + WINDOW_SIZE[0],
                                           y_off + WINDOW_SIZE[1]))
            new_image.thumbnail(OUT_SCALE_TO)
            for angle in ROTATIONS:
                out_fname = os.path.join(out_dir,
                                         '_'.join((base_id,
                                                   str(out_i),
                                                   kind)) + '.png')
                new_image.rotate(angle).save(out_fname)
                out_i += 1


def copy_one_augmented(src_dir, prefix, target_dir):
    augment_image_deterministic(os.path.join(src_dir,
                                             prefix + IN_SUFFIX),
                                target_dir)
    augment_image_deterministic(os.path.join(src_dir,
                                             prefix + OUT_SUFFIX),
                                target_dir)


def copy_all_augmented(src_dir, prefixes, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for prefix in prefixes:
        copy_one_augmented(src_dir, prefix, target_dir)


all_sample_names = list(set(os.path.basename(fname)[:-len(IN_SUFFIX)]
                            for fname in glob.glob('/notebook/data/4_inout_pairs/*' + IN_SUFFIX)))
random.shuffle(all_sample_names)

total_samples = len(all_sample_names)
train_number = int(total_samples * TRAIN_SIZE)
val_number = int(total_samples * VAL_SIZE)

train_prefixes = all_sample_names[:train_number]
val_prefixes = all_sample_names[train_number:train_number+val_number]
test_prefixes = all_sample_names[train_number+val_number:]

copy_all_augmented('/notebook/data/4_inout_pairs',
                   train_prefixes,
                   '/notebook/data/5_ready/train')
copy_all_augmented('/notebook/data/4_inout_pairs',
                   val_prefixes,
                   '/notebook/data/5_ready/val')
copy_all_augmented('/notebook/data/4_inout_pairs',
                   test_prefixes,
                   '/notebook/data/5_ready/test')