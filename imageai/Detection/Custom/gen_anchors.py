import random
import numpy as np

from imageai.Detection.Custom.voc import parse_voc_annotation


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def generateAnchors(train_annotation_folder, train_image_folder, train_cache_file, model_labels):

    print("Generating anchor boxes for training images and annotation...")
    num_anchors = 9

    train_imgs, train_labels = parse_voc_annotation(
        train_annotation_folder,
        train_image_folder,
        train_cache_file,
        model_labels
    )

    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:

        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/image['width']
            relative_h = (float(obj["ymax"]) - float(obj['ymin']))/image['height']
            annotation_dims.append(tuple(map(float, (relative_w,relative_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('Average IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    anchor_array = []
    reverse_anchor_array = []
    out_string = ""
    r = "anchors: ["
    for i in sorted_indices:
        anchor_array.append(int(anchors[i, 0] * 416))
        anchor_array.append(int(anchors[i, 1] * 416))

        out_string += str(int(anchors[i, 0] * 416)) + ',' + str(int(anchors[i, 1] * 416)) + ', '

    reverse_anchor_array.append(anchor_array[12:18])
    reverse_anchor_array.append(anchor_array[6:12])
    reverse_anchor_array.append(anchor_array[0:6])

    print("Anchor Boxes generated.")
    return anchor_array, reverse_anchor_array


