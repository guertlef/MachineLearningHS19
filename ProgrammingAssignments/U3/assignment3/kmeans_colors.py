from sklearn.cluster import KMeans
import numpy as np
from time import time
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin

def kmeans_colors(img, k, max_iter=100):
    """
    Performs k-means clusering on the pixel values of an image.
    Used for color-quantization/compression.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of color clusters to be computed

    Returns:
        img_cl:  The color quantized image of shape [h, w, 3]

    """

    img_cl = None

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the pixel values of the image img.     #
    #######################################################################
    w,h,d = img.shape
    assert d==3
    # Transform image to 2D numpy array
    img_array = img.reshape((w*h, d))
    
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=max_iter)
    kmeans = kmeans.fit(img_array)
    
    cluster_centers = kmeans.cluster_centers_[kmeans.labels_, :]
    img_cl = cluster_centers.reshape(img.shape).astype(int)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return img_cl
