import cv2
import numpy as np
import pickle


def resize(img):
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
    return img


def remove_edge(img):
    pos = np.nonzero(img)
    row = [max(pos[0]), min(pos[0])]
    column = [max(pos[1]), min(pos[1])]
    img = img[row[1]:row[0], column[1]:column[0]]
    return img


def binarization(img):
    img[np.where(img <= 199)] = 0
    img[np.where(img >= 200)] = 255
    img = 255 - img
    img = remove_edge(img)
    img = resize(img)
    return img


def rotation(img,angle):
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    rotate = cv2.getRotationMatrix2D((45, 45), angle, 1)
    img = cv2.warpAffine(img, rotate, (90, 90))
    img = remove_edge(img)
    return img


def add_erode_dilate(img, iter):
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((3, 3), np.uint8)
    if iter >= 0:
        out = cv2.dilate(img, kernel, iterations=iter)
    else:
        out = cv2.erode(img, kernel, iterations=-iter)
    return out


image_path = 'C:/myproject/testimage/'
image_path = 'myproject/input/'
character = ['backward', 'forward', 'up', 'down', 'left', 'right', 'entrance', 'exit']

# for p in range(len(character)):
#     random_angle = np.random.randn(10)
#     output_x = np.zeros((13*10*3, 50, 50), dtype=np.uint8)
#     m = 0
#     for i in range(13):
#         image = cv2.imread(image_path + character[p] + '/%d.jpg'%(i+1), cv2.IMREAD_GRAYSCALE)
#         image = binarization(image)
#         for a in range(10):
#             img1 = rotation(image, random_angle[a]*10)
#             for b in range(-1,2):
#                 img2 = add_erode_dilate(img1, b)
#                 img2 = resize(img2)
#                 output_x[m] = img2
#                 m = m + 1
#     output_y = [p] * 13 * 10 * 3
#     data = {'x': output_x, 'y': output_y, 'rows': 50, 'cols': 50}
#     pickle.dump(data, open("C:/myproject/outputimage/%s.pickle" % character[p], "wb"))

#image = cv2.imread(image_path + character[2] + '/%d.jpg'%(2), cv2.IMREAD_GRAYSCALE)
image = cv2.imread(image_path + character[2] + '.jpg', cv2.IMREAD_GRAYSCALE)
image = binarization(image)
image1 = add_erode_dilate(image, -1)
resized = add_erode_dilate(image, 1)
image = np.vstack((image1,resized))
cv2.imshow('output1', image)
cv2.waitKey(0)

