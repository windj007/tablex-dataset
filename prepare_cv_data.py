#!/usr/bin/env python

import os, glob, random
from train_test_augment import copy_all_augmented, copy_all_svg, IN_SUFFIX

CV_VAL_SIZE = 0.1

def cv_split(in_dir='/notebook/data/4_inout_pairs/', out_dir='/notebook/data/6_eval/', svg_dir='/notebook/data/3_prepared_images/', val_size=CV_VAL_SIZE, folds=5):
    all_sample_names = list(set(os.path.basename(fname)[:-len(IN_SUFFIX)]
                                for fname in glob.glob(os.path.join(in_dir, '*' + IN_SUFFIX))))
    random.shuffle(all_sample_names)

    total_samples = len(all_sample_names)
    block_size = len(all_sample_names) // folds
    for fold_i in range(folds):
        fold_start = fold_i * block_size
        fold_end = total_samples if fold_i == folds - 1 else (fold_start + block_size)

        all_train_names = all_sample_names[:fold_start] + all_sample_names[fold_end:]
        random.shuffle(all_train_names)
        val_split = int(len(all_train_names) * CV_VAL_SIZE)

        train_names = all_train_names[:-val_split]
        copy_all_augmented(in_dir,
                           train_names,
                           os.path.join(out_dir, str(fold_i), 'train'))

        val_names = all_train_names[-val_split:]
        copy_all_augmented(in_dir,
                           val_names,
                           os.path.join(out_dir, str(fold_i), 'val'))

        test_names = all_sample_names[fold_start:fold_end]
        copy_all_svg(svg_dir,
                     test_names,
                     os.path.join(out_dir, str(fold_i), 'test'))


if __name__ == '__main__':
    cv_split()
