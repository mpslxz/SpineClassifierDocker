import os
import sys
import numpy as np
import cv2
from extractFeatures import extractFeatures


def normalize(nd_array):
    return (nd_array - nd_array.mean())/nd_array.std()


def sigmoid(x):
    return 1.0/(1. + np.exp(-x))


def softmax(x):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def classify(file_name):
    cwd = os.path.dirname(__file__)
    params = np.load(os.path.join(cwd, 'ckpts', 'np_params.npy'))
    _, extension = os.path.splitext(file_name)
    if extension != '.npy':
        img = cv2.imread(file_name, 0).astype('float32')
    else:
        img = np.load(file_name)
    if img.ndim > 2:
        out = []
        for i in len(img):
            features = np.array(extractFeatures(normalize(img[i]))).reshape((1, -1))
            _y = np.dot(features, params[0]) + params[1]
            y = np.dot(sigmoid(_y), params[2]) + params[3]
            out.append(softmax(y))
        return np.array(out)
    features = np.array(extractFeatures(normalize(img))).reshape((1, -1))
    _y = np.dot(features, params[0]) + params[1]
    y = np.dot(sigmoid(_y), params[2]) + params[3]
    return softmax(y)


if __name__ == "__main__":
    assert os.path.exists(sys.argv[1])
    cwd = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(cwd, 'data')):
        os.makedirs(os.path.join(cwd, 'data'))
    print sys.argv[1]
    np.save(os.path.join(cwd, 'data', 'output.npy'), np.array(classify(sys.argv[1])))
