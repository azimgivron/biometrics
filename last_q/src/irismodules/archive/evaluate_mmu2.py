# ------------------------------------------------------------------------------
# 	Libraries
# ------------------------------------------------------------------------------
import os
from collections import defaultdict
from glob import glob
from itertools import repeat
from multiprocessing import Pool, cpu_count
from random import shuffle
from time import time

import numpy as np
from fnc.extractFeature import extractFeature
from fnc.matching import calHammingDist
from matplotlib import pyplot as plt
from tqdm import tqdm

# ------------------------------------------------------------------------------
#   Parameters
# ------------------------------------------------------------------------------
MMU2_DIR = "/home/antiaegis/Downloads/Iris-Recognition/MMU2"
EYELASHES_THRES = 80
N_IMAGES = 2


# ------------------------------------------------------------------------------
#   Pool function of extracting feature
# ------------------------------------------------------------------------------
def pool_func_extract_feature(args):
    im_filename, eyelashes_thres, use_multiprocess = args

    template, mask, im_filename = extractFeature(
        im_filename=im_filename,
        eyelashes_thres=eyelashes_thres,
        use_multiprocess=use_multiprocess,
    )
    return template, mask, im_filename


# ------------------------------------------------------------------------------
#   Pool function of calculating Hamming distance
# ------------------------------------------------------------------------------
def pool_func_calHammingDist(args):
    template1, mask1, template2, mask2 = args
    dist = calHammingDist(template1, mask1, template2, mask2)
    return dist


# ------------------------------------------------------------------------------
# 	Main execution
# ------------------------------------------------------------------------------
# Get identities of MMU2 dataset
identities = glob(os.path.join(MMU2_DIR, "**"))
identities = sorted([os.path.basename(identity) for identity in identities])
n_identities = len(identities)
print("Number of identities:", n_identities)


# Construct a dictionary of files
files_dict = {}
image_files = []
for identity in identities:
    # Because the 50th identity has only 5 images, skip it
    if identity == "50":
        continue

    files = glob(os.path.join(MMU2_DIR, identity, "*.bmp"))
    shuffle(files)
    files_dict[identity] = files[:N_IMAGES]
    # print("Identity %s: %d images" % (identity, len(files_dict[identity])))
    image_files += files[:N_IMAGES]

n_image_files = len(image_files)
print("Number of image files:", n_image_files)


# Extract features
args = zip(image_files, repeat(EYELASHES_THRES), repeat(False))
pools = Pool(processes=cpu_count())

start_time = time()
features = list(pools.map(pool_func_extract_feature, args))
finish_time = time()
print("Extraction time: %.3f [s]" % (finish_time - start_time))


# Calculate the distances
args = []
for i in range(n_image_files):
    for j in range(n_image_files):
        if i >= j:
            continue

        arg = (features[i][0], features[i][1], features[j][0], features[j][1])
        args.append(arg)
print("Number of pairs:", len(args))

start_time = time()
distances = pools.map(pool_func_calHammingDist, args)
finish_time = time()
print("Extraction time: %.3f [s]" % (finish_time - start_time))


# Construct a distance matrix
dist_mat = np.zeros([n_image_files, n_image_files])
k = 0
for i in range(n_image_files):
    for j in range(n_image_files):
        if i < j:
            dist_mat[i, j] = distances[k]
            k += 1
        elif i > j:
            dist_mat[i, j] = dist_mat[j, i]

np.save("dist_mat.npy", dist_mat)

plt.figure()
plt.imshow(dist_mat)
plt.show()
