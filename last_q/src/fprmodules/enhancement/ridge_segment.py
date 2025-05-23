# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 23:04:30 2016

@author: utkarsh, adapted to OpenCV by dvdm
"""


# RIDGESEGMENT - Normalises fingerprint image and segments ridge region
#
# Function identifies ridge regions of a fingerprint image and returns a
# mask identifying this region.  It also normalises the intensity values of
# the image so that the ridge regions have zero mean, unit standard
# deviation.
#
# This function breaks the image up into blocks of size blksze x blksze and
# evaluates the standard deviation in each region.  If the standard
# deviation is above the threshold it is deemed part of the fingerprint.
# Note that the image is normalised to have zero mean, unit standard
# deviation prior to performing this process so that the threshold you
# specify is relative to a unit standard deviation.
#
# Usage:   [normim, mask, maskind] = ridgesegment(im, blksze, thresh)
#
# Arguments:   im     - Fingerprint image to be segmented.
#              blksze - Block size over which the the standard
#                       deviation is determined (try a value of 16).
#              thresh - Threshold of standard deviation to decide if a
#                       block is a ridge region (Try a value 0.1 - 0.2)
#
# Returns:     normim - Image where the ridge regions are renormalised to
#                       have zero mean, unit standard deviation.
#              mask   - Mask indicating ridge-like regions of the image,
#                       0 for non ridge regions, 1 for ridge regions.
#              maskind - Vector of indices of locations within the mask.
#
# Suggested values for a 500dpi fingerprint image:
#
#   [normim, mask, maskind] = ridgesegment(im, 16, 0.1)
#
# See also: RIDGEORIENT, RIDGEFREQ, RIDGEFILTER

### REFERENCES

# Peter Kovesi
# School of Computer Science & Software Engineering
# The University of Western Australia
# pk at csse uwa edu au
# http://www.csse.uwa.edu.au/~pk


import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt


def normalise(img, mean, std):
    normed = (img - np.mean(img)) / (np.std(img))
    return normed


##
# def ridge_segment(im,blksze,thresh):
#
#    rows,cols = im.shape;
#
#    im = normalise(im,0,1);    # normalise to get zero mean and unit standard deviation
#
#
#    new_rows =  np.int(blksze * np.ceil((np.float(rows))/(np.float(blksze))))
#    new_cols =  np.int(blksze * np.ceil((np.float(cols))/(np.float(blksze))))
#
#    padded_img = np.zeros((new_rows,new_cols));
#    stddevim = np.zeros((new_rows,new_cols));
#
#    padded_img[0:rows][:,0:cols] = im;
#
#    for i in range(0,new_rows,blksze):
#        for j in range(0,new_cols,blksze):
#            block = padded_img[i:i+blksze][:,j:j+blksze];
#
#            stddevim[i:i+blksze][:,j:j+blksze] = np.std(block)*np.ones(block.shape)
#
#    stddevim = stddevim[0:rows][:,0:cols]
#
#    mask = stddevim > thresh;
#
#    mean_val = np.mean(im[mask]);
#
#    std_val = np.std(im[mask]);
#
#    normim = (im - mean_val)/(std_val);
#
#    return(normim,mask)


# dvdm: more efficient calculation of standard-deviation
# idea copied from
# https://stackoverflow.com/questions/28931265/calculating-variance-of-an-image-python-efficiently
#


def winStd(img, wlen):
    wmean, wsqrmean = (
        cv2.boxFilter(
            x, -1, (wlen, wlen), normalize=True, borderType=cv2.BORDER_REFLECT
        )
        for x in (img, img * img)
    )
    return np.sqrt(np.abs(wsqrmean - wmean * wmean))


def largest_connected_component(maskin):
    maskin = maskin.astype("uint8")
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        maskin, connectivity=4
    )
    # cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
    sizes = stats[:, cv2.CC_STAT_AREA]

    # stats[0], centroids[0] are for the background label. ignore
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    imgLCC = np.zeros(output.shape)
    # 	imgLCC[output == max_label] = 255
    imgLCC = output == max_label

    return imgLCC


def ridge_segment(im, blksze, thresh):
    rows, cols = im.shape

    # normalise to get zero mean and unit standard deviation
    im = normalise(im, 0, 1)

    # calculate local standard deviation
    stddevim = winStd(im, blksze)
    # 	print(stddevim)
    # threshold std image
    mask = stddevim > thresh
    # 	plt.imshow(mask)
    # 	plt.show()

    # fill holes
    mask = scipy.ndimage.binary_fill_holes(input=mask)
    # 	plt.imshow(mask)
    # 	plt.show()

    mask = largest_connected_component(mask)  # scipy.ndimage.label(mask)[0]
    # 	plt.imshow(mask)
    # 	plt.show()

    #
    mean_val = np.mean(im[mask])
    std_val = np.std(im[mask])
    normim = (im - mean_val) / (std_val)
    return (normim, mask)
