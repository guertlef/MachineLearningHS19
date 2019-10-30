from skimage.feature import hog
import numpy as np

def hog_features(X):
    """
    Extract HOG features from input images

    Args:
        X: Data matrix of shape [num_train, 577]

    Returns:
        hogs: Extracted hog features

    """
    
    hog_list = []
    
    for i in range(X.shape[0]):
        #######################################################################
        # TODO:                                                               #
        # Extract HOG features from each image and append them to the         #
        # hog_list                                                            #
        #                                                                     #
        # Hint: Make sure that you reshape the imput features to size (24,24) #
        #                                                                     #
        #######################################################################

        img = np.reshape(X[i,1:], [24,24], order='F')
        
        hog_feat = hog(img, orientations=6, pixels_per_cell=(4,4), 
                       cells_per_block=(1,1), block_norm='L2-Hys')
        
        hog_list.append(hog_feat)
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        
    hogs = np.stack(hog_list,axis=0)
    hogs = np.concatenate((np.ones((X.shape[0], 1)), np.reshape(hogs,(X.shape[0],-1))), axis=1)

    return hogs