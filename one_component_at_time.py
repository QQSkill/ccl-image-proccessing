#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import cv2
  
# read the image file
img = cv2.imread('./Yinyang.png', 2)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

def neighbor_value(i, j, img):
    left = img[i-1,j]
    # above
    above = img[i,j-1]
    neighbor_array = [left,above]
    return neighbor_array

def neighbor(i, j, img):
    left = [i-1, j]
    right = [i+1, j]
    top = [i, j-1]
    down = [i, j+1]
    top_left = [i-1, j-1]
    top_right = [i+1, j-1]
    down_left = [i-1, j+1]
    down_right = [i+1, j+1]
    neighbor_array = [left, right, top, down, top_left, top_right, down_left, down_right]
    result = []
    for idx in neighbor_array:
        if idx[0] >= img.shape[0] or idx[1] >= img.shape[1]:
            continue
        if img[i][j] == img[idx[0]][idx[1]]:
            result.append([idx[0], idx[1]])
    return result

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def labeling(image_org):
    size = image_org.shape
    m = size[0]  # rows
    n = size[1]  # columns

    print(size)

    """ for i in range(m):
        for j in range(n):
            if gray[i,j] > threshold:
                gray[i,j] = 0
                f.write(str(int(gray[i,j])))
            else:
                gray[i,j] = 1
                f.write(str(int(gray[i,j])))
        f.write('\n') """
    image = image_org
    #print(image)

    label = np.zeros([m,n], dtype='int')
    current = 0

    # link array
    queue = []
    id = 0 # link index also present object number

    # first pass
    for row in range(m):
        for column in range(n):
            if label[row][column] == 0:
                current += 1
                label[row][column] = current
                queue.append([row , column])
            while len(queue) != 0:
                last = queue.pop()
                neighbor_idx = neighbor(last[0], last[1], image)
                for idx in neighbor_idx:
                    if label[idx[0]][idx[1]] == 0:
                        label[idx[0]][idx[1]] = current
                        queue.append(idx)

    plt.imshow(label)
    plt.axis('off')
    plt.show()
    return label,image,id


label,image ,id = labeling(bw_img)
