import numpy as np
from sklearn.mixture import GaussianMixture

import time


def em_segmentation(img, k, max_iter=20):
    """
    Learns a MoG model using the EM-algorithm for image-segmentation.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of gaussians to be used

    Returns:
        label_img: A matrix of labels indicating the gaussian of size [h, w]

    """

    label_img = None

    #######################################################################
    # TODO:                                                               #
    # 1st: Augment the pixel features with their 2D coordinates to get    #
    #      features of the form RGBXY (see np.meshgrid)                   #
    # 2nd: Fit the MoG to the resulting data using                        #
    #      sklearn.mixture.GaussianMixture                                #
    # 3rd: Predict the assignment of the pixels to the gaussian and       #
    #      generate the label-image                                       #
    #######################################################################
    # Step 1:
    h,w,d = img.shape
    assert d == 3
    
    xv, yv = np.meshgrid(np.arange(0,w), np.arange(0,h))
    coords = np.stack((yv, xv), axis=2)
    img = np.concatenate((img, coords), axis=2)
    img = np.reshape(img, (h*w, d+2)) # dimension +2 to go from rgb to rgbxy
    
    # Step 2:
    mog = GaussianMixture(n_components=k, 
                          max_iter=max_iter,
                          covariance_type="full").fit(img)
    
    # Step 3:
    labels = mog.predict(img)
    means = np.delete(mog.means_, [d,d+1], axis=1).astype('uint8')

    tmp_img = np.take(means, labels, axis=0)
    label_img = np.reshape(tmp_img, (h, w, d))

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return label_img
