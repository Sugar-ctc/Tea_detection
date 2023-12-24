import glob
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def cas_ratio(box,cluster):
    ratios_of_box_cluster = box / cluster
    ratios_of_cluster_box = cluster / box
    ratios = np.concatenate([ratios_of_box_cluster, ratios_of_cluster_box], axis = -1)

    return np.max(ratios, -1)

def avg_ratio(box,cluster):
    return np.mean([np.min(cas_ratio(box[i],cluster)) for i in range(box.shape[0])])

def kmeans(box,k):
    #-------------------------------------------------------------#
    #   total boxes 
    #-------------------------------------------------------------#
    row = box.shape[0]
    
    #-------------------------------------------------------------#
    #   The location of each point in each box
    #-------------------------------------------------------------#
    distance = np.empty((row,k))
    
    #-------------------------------------------------------------#
    #   Clustering position
    #-------------------------------------------------------------#
    last_clu = np.zeros((row,))

    np.random.seed()

    #-------------------------------------------------------------#
    #   5 clusters are randomly selected as clustering centers
    #-------------------------------------------------------------#
    cluster = box[np.random.choice(row,k,replace = False)]

    iter = 0
    while True:
        #-------------------------------------------------------------#
        #   Calculate the ratio of width to height between the current frame and the prior frame
        #-------------------------------------------------------------#
        for i in range(row):
            distance[i] = cas_ratio(box[i],cluster)
        
        #-------------------------------------------------------------#
        #   Fetch minimum
        #-------------------------------------------------------------#
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        #-------------------------------------------------------------#
        #   Find the midpoint of each class
        #-------------------------------------------------------------#
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near
        if iter % 5 == 0:
            print('iter: {:d}. avg_ratio:{:.2f}'.format(iter, avg_ratio(box,cluster)))
        iter += 1

    return cluster, near

def load_data(path):
    data = []
    #-------------------------------------------------------------#
    #   Look for box for every piece of xml
    #-------------------------------------------------------------#
    for xml_file in tqdm(glob.glob('{}/*xml'.format(path))):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        #-------------------------------------------------------------#
        #   Get its width and height for each target
        #-------------------------------------------------------------#
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # Get width and height
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)

if __name__ == '__main__':
    np.random.seed(0)
    
    input_shape = [640, 640]
    anchors_num = 9
    #-------------------------------------------------------------#
    #   To load the dataset, you can use the xml of VOC
    #-------------------------------------------------------------#
    path        = 'VOCdevkit/VOC2007/Annotations'
    
    #-------------------------------------------------------------#
    #   Load all the xml
    #   The storage format is width,height converted to scale
    #-------------------------------------------------------------#
    print('Load xmls.')
    data = load_data(path)
    print('Load xmls done.')
    
    #-------------------------------------------------------------#
    #   Using k clustering algorithm
    #-------------------------------------------------------------#
    print('K-means boxes.')
    cluster, near   = kmeans(data, anchors_num)
    print('K-means boxes done.')
    data            = data * np.array([input_shape[1], input_shape[0]])
    cluster         = cluster * np.array([input_shape[1], input_shape[0]])

    #-------------------------------------------------------------#
    #   draw
    #-------------------------------------------------------------#
    for j in range(anchors_num):
        plt.scatter(data[near == j][:,0], data[near == j][:,1])
        plt.scatter(cluster[j][0], cluster[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()
    print('Save kmeans_for_anchors.jpg in root dir.')

    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print('avg_ratio:{:.2f}'.format(avg_ratio(data, cluster)))
    print(cluster)

    f = open("yolo_anchors.txt", 'w')
    row = np.shape(cluster)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(x_y)
    f.close()
