import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def convolve(img, filter):
    rows, cols = img.shape
    convolved_img = np.zeros((rows - filter.shape[0] + 1, cols - filter.shape[1] + 1))
    for row in range(convolved_img.shape[0]):
        for col in range(convolved_img.shape[1]):
            convolved_img[row, col] = np.sum(
                np.multiply(
                    img[row : row + filter.shape[0], col : col + filter.shape[1]],
                    filter,
                )
            )

    return convolved_img


def image_hist(image, save=False, fname=""):
    if len(image.shape) == 3:
        # calculate mean value from RGB channels and flatten to 1D array
        vals = image.flatten()
    else:
        vals = image.flatten()

    fig, ax = plt.subplots(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax1.set_xlim([-5, 300])
    ax1.hist(vals, bins=range(256))

    ax2 = fig.add_subplot(122)
    ax2.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax2.axis("off")

    ax.axis("off")
    if save:
        plt.savefig(fname)
    plt.show()


def draw_mask(circles, image):
    image_copy = image.copy()
    # Print the output
    (x, y, r, v) = circles[0]
    (x1, y1, r1, v1) = circles[1]

    cv2.circle(image_copy, (int(y), int(x)), int(r), (255, 0, 0), 2)
    cv2.circle(image_copy, (int(y), int(x)), int(r1), (255, 0, 0), 2)

    gray_copy = image.copy()
    cv2.circle(gray_copy, (int(y) * 5, int(x) * 5), int(r) * 5, (255, 0, 0), 2)
    cv2.circle(gray_copy, (int(y) * 5, int(x) * 5), int(r1) * 5, (255, 0, 0), 2)

    gray_copy1 = image.copy()
    # draw masks
    mask1 = np.zeros_like(gray_copy1)
    mask1 = cv2.circle(
        mask1,
        (int(circles[0][1] * 5), int(circles[0][0]) * 5),
        int(circles[0][2]) * 5,
        (255, 255, 255),
        -1,
    )
    mask2 = np.zeros_like(gray_copy1)
    mask2 = cv2.circle(
        mask2,
        (int(circles[0][1] * 5), int(circles[0][0]) * 5),
        int(circles[1][2]) * 5,
        (255, 255, 255),
        -1,
    )
    mask = cv2.subtract(mask2, mask1)
    masked_resized = cv2.bitwise_and(gray_copy1, mask)
    return masked_resized, gray_copy


def resize(img, percent):
    print("image.shape : {}".format(img.shape))
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    strongest_i, strongest_j = np.where(resized >= 200)
    resized[strongest_i, strongest_j] = 0

    print("resized.shape : {}".format(resized.shape))
    return resized


# Edge detection functions
def sobel_filters(img):
    edge_threshold = 7
    Hx = np.array([[1, 0, -1], [edge_threshold, 0, -edge_threshold], [1, 0, -1]])
    Hy = np.array([[1, edge_threshold, 1], [0, 0, 0], [-1, -edge_threshold, -1]])

    Gx = convolve(img, Hx)
    Gy = convolve(img, Hy)
    G = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))

    strongest_i, strongest_j = np.where(G >= 200)
    G[strongest_i, strongest_j] = 0

    theta = np.arctan2(Gy, Gx)
    return (G, theta)


def non_max_suppression(img, D):
    rows = img.shape[0]
    cols = img.shape[1]
    non_max = np.zeros((rows, cols), dtype=np.int32)
    angle = D * 180.0 / np.pi
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            try:
                q = []
                if ((angle[i, j] < 22.5) and (-22.5 <= angle[i, j])) or (
                    (angle[i, j] < -157.5) and (157.5 <= angle[i, j])
                ):  # horizental
                    q = [
                        img[i, j + 1],
                        img[i, j + 2],
                        img[i - 1, j + 2],
                        img[i + 1, j + 2],
                        img[i, j - 1],
                        img[i, j - 2],
                        img[i - 1, j - 2],
                        img[i + 1, j - 2],
                    ]
                elif ((angle[i, j] < 67.5) and (22.5 <= angle[i, j])) or (
                    (angle[i, j] < -112.5) and (-157.5 <= angle[i, j])
                ):  # 45degree
                    q = [
                        img[i - 1, j + 1],
                        img[i - 1, j + 2],
                        img[i - 2, j + 1],
                        img[i - 2, j + 2],
                        img[i + 1, j - 1],
                        img[i + 1, j - 2],
                        img[i + 2, j - 1],
                        img[i + 2, j - 2],
                    ]
                elif ((angle[i, j] < 112.5) and (67.5 <= angle[i, j])) or (
                    (angle[i, j] < -67.5) and (-112.5 <= angle[i, j])
                ):  # vertical
                    q = [
                        img[i - 1, j],
                        img[i - 2, j],
                        img[i - 2, j + 1],
                        img[i - 2, j - 1],
                        img[i + 1, j],
                        img[i + 2, j],
                        img[i + 2, j + 1],
                        img[i + 2, j - 1],
                    ]
                else:  # 135degree
                    q = [
                        img[i + 1, j + 1],
                        img[i + 1, j + 2],
                        img[i + 2, j + 1],
                        img[i + 2, j + 2],
                        img[i - 1, j - 1],
                        img[i - 1, j - 2],
                        img[i - 2, j - 1],
                        img[i - 2, j - 2],
                    ]
                q.sort()
                if img[i, j] >= q[-3]:
                    non_max[i, j] = img[i, j]
                else:
                    non_max[i, j] = 0
            except IndexError as e:
                pass
    return non_max


def threshold_img(img):
    highThreshold = img.max() * 0.1
    img = img.astype("uint8")

    otsu_threshold_val, ret_matrix = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    lowThreshold = otsu_threshold_val
    rows, cols = img.shape
    res = np.zeros((rows, cols), dtype=np.int32)

    strong = np.int32(255)

    strongest_i, strongest_j = np.where(img == 255)
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where((img < lowThreshold))
    mid_i, mid_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[mid_i, mid_j] = 0
    res[zeros_i, zeros_j] = 0
    res[strongest_i, strongest_j] = 0
    return res


def hysteresis(img):
    rows, cols = img.shape

    pixel_threshold = 2

    weak = np.int32(75)
    strong = np.int32(255)

    for i in range(1, rows - pixel_threshold - 1):
        for j in range(1, cols - pixel_threshold - 1):
            if img[i, j] == weak:
                for k in range(-pixel_threshold, pixel_threshold + 1):
                    for q in range(-pixel_threshold, pixel_threshold + 1):
                        if img[i + k, j + q] == strong:
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
    return img
