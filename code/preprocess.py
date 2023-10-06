import cv2 as cv
import numpy as np


# def show_image():
#     for i in range(len(image_name)):
#         img = cv.imread(image_path + image_name[i])
#         cv.imshow('input_image', img)
#         cv.waitKey(0)
#     cv.destroyAllWindows()


def de_noise(img, median_p, gauss_p):
    img = cv.medianBlur(img, median_p)
    img = cv.blur(img, (gauss_p, gauss_p))
    return img


def color_quantization(img, color_n):
    Z = img.reshape((-1, 1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, color_n, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    return res2


def remove_texture(img,block_size):
    kernel = np.ones((block_size, block_size), np.uint8)
    # erosion = cv.erode(img, kernel, iterations=1)
    # dilation = cv.dilate(erosion, kernel, iterations=1)
    dilation = cv.dilate(img, kernel, iterations=1)
    erosion = cv.erode(dilation, kernel, iterations=1)
    return erosion


def binarization(img):
    w_b = []
    w_b.extend(np.hstack((img[:, 0], img[:, img.shape[1] - 1], img[0, :], img[img.shape[0] - 1, :])))
    bg_c = np.argmax(np.bincount(w_b))
    img[np.where(img != bg_c)] = 255
    img[np.where(img == bg_c)] = 0
    return img


def remove_edge(img):
    pos = np.nonzero(img)
    row = [max(pos[0]),min(pos[0])]
    column = [max(pos[1]),min(pos[1])]
    img = img[row[1]:row[0],column[1]:column[0]]
    return img


def main(img,size):
    img = de_noise(img,5,11)
    img = color_quantization(img, 2)
    img = binarization(img)
    # img = remove_texture(img, 7)
    img = remove_edge(img)
    img = cv.resize(img, size, interpolation=cv.INTER_LINEAR)
    return img

