from importlib import import_module
from getopt import getopt
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pprint
import sys
import os
import cv2
import math
import shutil
import re
from easydict import EasyDict as edict

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *

mapping_file = '/home/garrick/Desktop/detective/data/kitti_split1/devkit/mapping/train_mapping.txt'
rand_file = '/home/garrick/Desktop/detective/data/kitti_split1/devkit/mapping/train_rand.txt'
ids_file = '/home/garrick/Desktop/detective/data/kitti_split1/val.txt'
mapping = []
rand_map = []

with_tracklets = ['2011_09_26_drive_0086_sync',
                  '2011_09_26_drive_0064_sync',
                  '2011_09_26_drive_0070_sync',
                  '2011_09_26_drive_0022_sync',
                  '2011_09_26_drive_0039_sync',
                  '2011_09_26_drive_0032_sync',
                  '2011_09_26_drive_0014_sync',
                  '2011_09_26_drive_0009_sync',
                  '2011_09_26_drive_0023_sync',
                  '2011_09_26_drive_0052_sync',
                  '2011_09_26_drive_0093_sync',
                  '2011_09_26_drive_0002_sync',
                  '2011_09_26_drive_0017_sync',]


with_tracklets = ['2011_09_26_drive_0046_sync',
                 '2011_09_26_drive_0056_sync',
                 '2011_09_26_drive_0036_sync',
                 '2011_09_26_drive_0018_sync',
                 '2011_09_26_drive_0027_sync',
                 '2011_09_26_drive_0028_sync',
                 '2011_09_26_drive_0051_sync',
                 '2011_09_26_drive_0019_sync',
                 '2011_09_26_drive_0061_sync',
                 '2011_09_26_drive_0087_sync',
                 '2011_09_26_drive_0035_sync',
                 '2011_09_26_drive_0057_sync',
                 '2011_09_26_drive_0059_sync',
                 '2011_09_26_drive_0091_sync',
                 '2011_09_26_drive_0001_sync',
                 '2011_09_26_drive_0084_sync',
                 '2011_09_26_drive_0015_sync',
                 '2011_09_26_drive_0029_sync',
                 '2011_09_26_drive_0011_sync',
                 '2011_09_26_drive_0020_sync',
                 '2011_09_26_drive_0013_sync',
                 '2011_09_26_drive_0005_sync',
                 '2011_09_26_drive_0060_sync',
                 '2011_09_26_drive_0048_sync',
                 '2011_09_26_drive_0079_sync',]

# read mapping
text_file = open(mapping_file, 'r')

for line in text_file:

    # 2011_09_26 2011_09_26_drive_0005_sync 0000000109
    parsed = re.search('(\S+)\s+(\S+)\s+(\S+)', line)

    if parsed is not None:

        date = str(parsed[1])
        seq = str(parsed[2])
        id = str(parsed[3])
        mapping.append([seq, id])

text_file.close()

# read rand
text_file = open(rand_file, 'r')

for line in text_file:
    parsed = re.findall('(\d+)', line)
    for p in parsed:
        rand_map.append(int(p))

text_file.close()

text_file = open(ids_file, 'r')

seqs_used = []
# compute total sequences available
for rand in rand_map:
    if not mapping[rand-1][0] in seqs_used:
        seqs_used.append(mapping[rand-1][0])

total_max = len(seqs_used)

im_count = 0
tr_count = 0

# compute sequences used!
seqs_used = []
for line in text_file:

    parsed = re.search('(\d+)', line)

    if parsed is not None:
        id = int(parsed[0])

        im_count += 1
        if mapping[rand_map[id]-1][0] in with_tracklets:
            tr_count += 1

        if not mapping[rand_map[id]-1][0] in seqs_used:
            seqs_used.append(mapping[rand_map[id]-1][0])
            print('\'{}\','.format(mapping[rand_map[id]][0]))

actual_used = len(seqs_used)

print('with tracking? {}/{}, {}'.format(tr_count, im_count, tr_count/im_count))

#print(seqs_used)
text_file.close()

print('{}/{} seqs used'.format(actual_used, total_max))
