import numpy as np
import hadamard
import wedge

def buildFilters():
    hMats = []
    hMats.append(hadamard.walsh(128))
    hMats.append(hadamard.walsh(64))
    hMats.append(hadamard.walsh(32))

    wFilters = []
    wFilters.append(wedge.wedge(np.zeros((128,128)), 10, 1))
    wFilters.append(wedge.wedge(np.zeros((128,128)), 10, 2))
    W = 1 - wFilters[-1] - wFilters[-2]
    W[0,0] = 0
    wFilters.append(W)

    wFilters.append(wedge.wedge(np.zeros((64,64)), 10, 1))
    wFilters.append(wedge.wedge(np.zeros((64,64)), 10, 2))
    W = 1 - wFilters[-1] - wFilters[-2]
    W[0,0] = 0
    wFilters.append(W)

    wFilters.append(wedge.wedge(np.zeros((32,32)), 10, 1))
    wFilters.append(wedge.wedge(np.zeros((32,32)), 10, 2))
    W = 1 - wFilters[-1] - wFilters[-2]
    W[0,0] = 0
    wFilters.append(W)

    return (hMats, wFilters)
