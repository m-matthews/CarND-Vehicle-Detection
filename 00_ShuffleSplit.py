# -*- coding: utf-8 -*-
"""
Vehicle Detection - Test GroupShuffleSplit

Author: Michael Matthews
"""

#import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit

def sampleimage(chanset):
    """Create a sample blank image with a fixed color.
    
    chanset: [0] = Red, [1] = Green, [2] = Blue, [1,2] = Cyan ...
    
    returns image 64 pixels square.
    """
    image = np.zeros((64,64,3), np.uint8)
    image[:,:,chanset] = 255
    return image

if __name__ == '__main__':
    # Demonstrate shuffle differences with grouping.
    X = []
    y = []
    g = []
    for count, chanset, group in [(3, [0], 'red'), (3, [1], 'green'),
                                  (3, [2], 'blue'), (1, [1, 2], 'cyan'),
                                  (1, [0, 1], 'yellow'), (1, (0, 2), 'purple')]:
        for i in range(count):
            X.append(sampleimage(chanset))
            y.append(1)
            g.append(group)
    
    ss = ShuffleSplit(n_splits=1, test_size=0.3)
    sss = ss.split(X, y)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
    gsss = gss.split(X, y, groups=np.array(g))

    for split, title in ((sss, 'ShuffleSplit'), (gsss, 'GroupShuffleSplit')):
        for train, test in split:
            X_train = [X[i] for i in train]
            y_train = [y[i] for i in train]
            X_test = [X[i] for i in test]
            y_test = [y[i] for i in test]

        fig, ax = plt.subplots(2, len(X_train), figsize=(6,3))
        fig.suptitle(title)
        for i in range(len(X_train)):
            ax[0][i].imshow(X_train[i])
            ax[0][i].axis('off')

            if i<len(X_test):
                ax[1][i].imshow(X_test[i])
            ax[1][i].axis('off')

            if i==0:
                ax[0][i].set_title('Train')
                ax[1][i].set_title('Test')

        plt.savefig("./output_images/split_{}.png".format(title))
