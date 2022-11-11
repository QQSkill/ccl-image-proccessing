import matplotlib.pyplot as plt
import numpy as np
import cv2
  
# read the image file
img = cv2.imread('./mediacal.png', 2)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# neighbor and current_poit=image[i,j]
def neighbor_label(i,j,label):
    # left
    left = label[i-1,j]
    # above
    above = label[i,j-1]
    neighbor_array = [left,above]
    return neighbor_array

def neighbor_value(i, j, img):
    left = img[i-1,j]
    # above
    above = img[i,j-1]
    neighbor_array = [left,above]
    return neighbor_array

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def labeling(image_org):

    #image_org = np.reshape(image_org,(:,:,1))
    size = image_org.shape  # Gray = R*0.299 + G*0.587 + B*0.114 ; size: 1104x1399
    m = size[0]  # rows
    n = size[1]  # columns

    print(size)

    # size = gray.shape # Gray = R*0.299 + G*0.587 + B*0.114 ; size: 1104x1399
    # m = size[0] #rows
    # n = size[1] #columns
    #print(m,n)
    #print(image_org.shape)
    
    # gray2binary
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
    new = 0

    # link array
    link = []
    id = 0 # link index also present object number

    # first pass
    for row in range(m):
        for column in range(n):
            # no objec
                #print(image[row, column], row + 1, column + 1,label[row, column])
            # object
            # check neighbor label
                #print(image[row, column], row + 1, column + 1)
            
            current_neighbor = neighbor_label(row,column,label)
            value_neighbor = neighbor_value(row, column, image)
            # current is new label
            if image[row, column] != value_neighbor[0] and image[row, column] != value_neighbor[1]:
                new += 1
                label[row, column] = new
            elif image[row, column] != value_neighbor[0] and image[row, column] == value_neighbor[1]:
                label[row, column] = current_neighbor[1]
            elif image[row, column] == value_neighbor[0] and image[row, column] != value_neighbor[1]:
                label[row, column] = current_neighbor[0]
            elif image[row, column] == value_neighbor[0] and image[row, column] == value_neighbor[1]:
                if current_neighbor[0] == current_neighbor[1]:
                    label[row, column] = current_neighbor[0]
                else:
                    label[row,column] = np.min(current_neighbor)
                    #print(id)
                    if id == 0:
                        link.append(set(current_neighbor))
                        id += 1
                        #print(link)
                    else:
                        check = 0
                        for k in range(id) :
                            tmp = set(link[k]).intersection(set(current_neighbor))
                            #print(k,link[k],current_neighbor,len(tmp))
                            if len(tmp) != 0 :
                                link[k] = set(link[k]).union(current_neighbor)
                                check = check + 1
                                #print(link)
                        if check == 0:
                            id += 1
                            link.append(set(current_neighbor))
    # second pass
    for row in range(m):
        for column in range(n):
            for x in range(id):
                if (label[row, column] in link[x]):
                    label[row, column] = min(link[x])
    for row in range(m):
        for column in range(n):
            for x in range(id):
                if (label[row, column] == min(link[x])):
                    label[row, column] = x+1
    print(label)
    plt.imshow(label)
    plt.axis('off')
    plt.show()
    return label,image,id


label,image ,id = labeling(bw_img)
