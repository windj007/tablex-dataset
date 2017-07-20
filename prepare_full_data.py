#!/usr/bin/env python

from train_test_augment import shuffle_split

if __name__ == '__main__':
    shuffle_split(out_dir='/notebook/data/7_full/', train_size=0.9, val_size=0.1)
