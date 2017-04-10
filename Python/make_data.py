import os
import cv2
import numpy as np
from extractFeatures import extractFeatures
from theano_ops.utils import normalize

if __name__ == "__main__":
    _ind = 1 + np.random.permutation(575)
    train_ind = _ind[:500]
    test_ind = _ind[500:]

    lam = []
    no_lam = []
    for i in train_ind:
        img = cv2.imread('train_data/newLam/img_{}.jpg'.format(i), 0)
        lam += [np.array(extractFeatures(normalize(img))).reshape((1, -1))]
        lam += [np.array(extractFeatures(normalize(np.fliplr(img)))).reshape((1, -1))]
        
        img = cv2.imread('train_data/notLamina/IMG_{}.jpg'.format(i), 0)
        no_lam += [np.array(extractFeatures(normalize(img))).reshape((1, -1))]
        no_lam += [np.array(extractFeatures(normalize(np.fliplr(img)))).reshape((1, -1))]
    train_x = np.vstack((np.array(lam).squeeze(), np.array(no_lam).squeeze()))
    train_y = np.hstack((np.ones((1, len(lam))), np.zeros((1, len(no_lam)))))

    print train_x.shape

    np.save('train_x', train_x)
    np.save('train_y', train_y)
    
    lam = []
    no_lam = []
    for i in test_ind:
        img = cv2.imread('train_data/newLam/img_{}.jpg'.format(i), 0)
        lam += [np.array(extractFeatures(normalize(img))).reshape((1, -1))]
        lam += [np.array(extractFeatures(normalize(np.fliplr(img)))).reshape((1, -1))]
        
        img = cv2.imread('train_data/notLamina/IMG_{}.jpg'.format(i), 0)
        no_lam += [np.array(extractFeatures(normalize(img))).reshape((1, -1))]
        no_lam += [np.array(extractFeatures(normalize(np.fliplr(img)))).reshape((1, -1))]
    test_x = np.vstack((np.array(lam).squeeze(), np.array(no_lam).squeeze()))
    test_y = np.hstack((np.ones((1, len(lam))), np.zeros((1, len(no_lam)))))
    
    np.save('test_x', test_x)
    np.save('test_y', test_y)
