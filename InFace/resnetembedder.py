import numpy as np
import tensorflow
import cv2
base_model = tensorflow.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',pooling='max', classifier_activation="softmax",name="resnet152v2")
def siamese_network(triplets):
    outcome = False
    def recognition(distances, threshold=0.5):
        recognized = []
        positive_count = 0
        negative_count = 0

        for d_a_p, d_a_n in distances:
            print(d_a_p, d_a_n)
            if d_a_p < d_a_n and d_a_p < threshold:
                recognized.append("Recognized")
                positive_count += 1
            else:
                recognized.append("Not Recognized")
                negative_count += 1

        return recognized, positive_count, negative_count
    def calculate_distances(anchors_emb, positives_emb, negatives_emb):
        distances = []
        for a_emb, p_emb, n_emb in zip(anchors_emb, positives_emb, negatives_emb):
            d_a_p = np.linalg.norm(a_emb - p_emb)  # Distance between anchor and positive
            d_a_n = np.linalg.norm(a_emb - n_emb)  # Distance between anchor and negative
            distances.append((d_a_p, d_a_n))
        return distances
    def convert_triplets_to_seperate():
         anchors, positives, negatives = zip(*triplets)
         # Convert to lists (if you need lists instead of tuples)
         anchors = np.array(list(anchors))
         positives = np.array(list(positives))
         negatives = np.array(list(negatives))
         return anchors, positives, negatives
    def embedding(anchors, positives, negatives):
        # Ensure that all inputs are valid
        if len(anchors) == 0 or len(positives) == 0 or len(negatives) == 0:
            raise ValueError("One or more input arrays are empty.")
            # Normalize images to the range [0, 1]
        anchors = np.array([img / 255.0 for img in anchors if isinstance(img, np.ndarray)])
        positives = np.array([img / 255.0 for img in positives if isinstance(img, np.ndarray)])
        negatives = np.array([img / 255.0 for img in negatives if isinstance(img, np.ndarray)])
        #inception resnet only works with size of 299 X 299
        anchors = np.array([cv2.resize(img, (299, 299)) for img in anchors if isinstance(img, np.ndarray)])
        positives = np.array([cv2.resize(img, (299, 299)) for img in positives if isinstance(img, np.ndarray)])
        negatives = np.array([cv2.resize(img, (299, 299)) for img in negatives if isinstance(img, np.ndarray)])
        # Check shapes after resizing
        print("Shape of anchors:", anchors.shape)
        print("Shape of positives:", positives.shape)
        print("Shape of negatives:", negatives.shape)

        #preprocess first and then embed the layer
        anchors = tensorflow.keras.applications.inception_resnet_v2.preprocess_input(anchors)
        positives = tensorflow.keras.applications.inception_resnet_v2.preprocess_input(positives)
        negatives = tensorflow.keras.applications.inception_resnet_v2.preprocess_input(negatives)
        anchor_embeddings = base_model.predict(anchors)
        positive_embeddings = base_model.predict(positives)
        negative_embeddings = base_model.predict(negatives)
        return anchor_embeddings, positive_embeddings, negative_embeddings
    anchors, positives, negatives = convert_triplets_to_seperate()
    anchor_embedings,positive_embedings,negative_embedings = embedding(anchors,positives,negatives)
    distance = calculate_distances(anchor_embedings,positive_embedings,negative_embedings)
    recognized,p,n = recognition(distance,threshold = 0.5)
    print(recognized)
    if p>n:
        outcome = True
    else:
        outcome = False
    return outcome








