import os
from recognize import recognize_face
from PIL import Image
import numpy as np
def create_triplet_path(anchor,labels,path):
    triplets_path = []
    images_file = labels
    Anchor = anchor
    for image_file in images_file:
        positive_label = image_file
        print(f"positive label {positive_label}")
        positive_label_path = os.path.join(path,positive_label)
        print(f"positive_label_path{positive_label_path}")
        p = 0
        while p<len(os.listdir(positive_label_path)):
            positive = os.listdir(positive_label_path)[p]
            print(f"positive image {positive}")
            positive_image_path = os.path.join(positive_label_path,positive)
            print(f"positive_image_path{positive_image_path}")
            negative_index = np.random.choice([x for x in range(len(images_file)) if x != positive_label])
            negative_label = images_file[negative_index]
            print(f"negative label {negative_label}")
            negative_label_path = os.path.join(path,negative_label)
            print(f"negative_label_path{negative_label_path}")
            negative = os.listdir(negative_label_path)[np.random.choice(len(os.listdir(negative_label_path)))]
            print(f"negative image {negative}")
            negative_image_path = os.path.join(negative_label_path,negative)
            print(f"negative_image_path{negative_image_path}")
            triplets_path.append([Anchor,positive_image_path,negative_image_path])
            p = p + 1

    return triplets_path


def path_extracter(triplets_path):
    triplets = []
    for triplet in triplets_path:
        anchor = triplet[0]
        positive_image = np.array(Image.open(triplet[1]))
        negative_image = np.array(Image.open(triplet[2]))
        triplets.append([anchor,positive_image,negative_image])
    return triplets






