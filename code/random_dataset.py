import cv2
import numpy as np
import pickle
import random

image_path = 'C:/myproject/outputimage/'
image_path = 'myproject/outputimage/'
character = ['backward', 'forward', 'up', 'down', 'left', 'right', 'entrance', 'exit']


backward = pickle.load(open(image_path + character[0] + '.pickle', 'rb'))
forward = pickle.load(open(image_path + character[1] + '.pickle', 'rb'))
up = pickle.load(open(image_path + character[2] + '.pickle', 'rb'))
down = pickle.load(open(image_path + character[3] + '.pickle', 'rb'))
left = pickle.load(open(image_path + character[4] + '.pickle', 'rb'))
right = pickle.load(open(image_path + character[5] + '.pickle', 'rb'))
entrance = pickle.load(open(image_path + character[6] + '.pickle', 'rb'))
exits = pickle.load(open(image_path + character[7] + '.pickle', 'rb'))


connect_setx = np.concatenate((backward['x'], forward['x'], up['x'], down['x'], left['x'], right['x'], entrance['x'], exits['x']), axis=0)
connect_sety = np.concatenate((backward['y'], forward['y'], up['y'], down['y'], left['y'], right['y'], entrance['y'], exits['y']))
rand = np.arange(len(connect_setx))
random.shuffle(rand)
rand_set = {'x': connect_setx[rand], 'y': connect_sety[rand]}
# for i in range(20):
#     img = rand_set['x'][i]
#     cv2.imshow('GrayImage', img)
#     cv2.waitKey()
#     print(rand_set['y'][i])
print(len(rand_set['y']))   # 3120
#pickle.dump(rand_set, open("C:/myproject/outputimage/rand_set.pickle", "wb"))

