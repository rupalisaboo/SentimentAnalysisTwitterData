import numpy as np
from scipy.linalg import eigh
import h5py


class _CCABase(object):
    def __init__(self, numCV=None, reg=None, regs=None, numCC=None, numCCs=None, kernelcca=True, ktype=None,
                 verbose=False, select=0.2, cutoff=1e-15):
        self.numCV = numCV
        self.reg = reg
        self.regs = regs
        self.numCC = numCC
        self.numCCs = numCCs
        self.kernelcca = kernelcca
        self.ktype = ktype
        self.cutoff = cutoff
        self.select = select
        if self.kernelcca and self.ktype == None:
            self.ktype = "linear"
        self.verbose = verbose

    def train(self, data):
        nT = data[0].shape[0]
        if self.verbose:
            if self.kernelcca:
                print("Training CCA, %s kernel, regularization = %0.4f, %d components" % (
                self.ktype, self.reg, self.numCC))
            else:
                print("Training CCA, regularization = %0.4f, %d components" % (self.reg, self.numCC))
        comps = kcca(data, self.reg, self.numCC, kernelcca=self.kernelcca, ktype=self.ktype)
        self.cancorrs, self.ws, self.comps = recon(data, comps, kernelcca=self.kernelcca)
        if len(data) == 2:
            self.cancorrs = self.cancorrs[np.nonzero(self.cancorrs)]
        return self


'''Attributes:
        reg - regularization parameters. Default is 0.1.
        numCC - number of canonical dimensions to keep. Default is 10.
        kernelcca - True if using a kernel (default), False if not kernelized.
        ktype - type of kernel if kernelcca == True (linear or gaussian). Default is linear.
        verbose - True is default

    Results:
        ws - canonical weights
        comps - canonical components
        cancorrs - correlations of the canonical components on the training dataset
        corrs - correlations on the validation dataset
        preds - predictions on the validation dataset
        ev - explained variance for each canonical dimension
'''

def kcca(data, reg=0., numCC=None, kernelcca=True, ktype="linear", returncorrs=False):
    '''Set up and solve the eigenproblem for the data in kernel and specified reg
    '''
    kernel = [_make_kernel(d, ktype=ktype) for d in data]
    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[1] for k in kernel]) if numCC is None else numCC

    # Get the kernel auto- and cross-covariance matrices
    crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]
    # Allocate LH and RH:
    LH = np.zeros((np.sum(nFs), np.sum(nFs)))
    RH = np.zeros((np.sum(nFs), np.sum(nFs)))

    # Fill the left and right sides of the eigenvalue problem
    for i in range(len(kernel)):
        RH[int(np.sum(nFs[:i])):int(np.sum(nFs[:i + 1])), int(np.sum(nFs[:i])):int(np.sum(nFs[:i + 1]))] = crosscovs[
                                                                                                               i * (len(
                                                                                                                   kernel) + 1)] + reg * np.eye(
            nFs[i])
        for j in range(len(kernel)):
            if i != j:
                LH[int(np.sum(nFs[:i])):int(np.sum(nFs[:i + 1])), int(np.sum(nFs[:j])):int(np.sum(nFs[:j + 1]))] = \
                crosscovs[len(kernel) * j + i]

    LH = (LH + LH.T) / 2.
    RH = (RH + RH.T) / 2.

    r, Vs = eigh(LH, RH)

    comp = []
    for i in range(len(kernel)):
        comp.append(Vs[int(np.sum(nFs[:i])):int(np.sum(nFs[:i + 1]))])
    tcorrs = recon(data, comp, corronly=True, kernelcca=kernelcca)
    tc = [t[0, 1] for t in tcorrs]
    i = np.argsort(tc)[::-1]
    comp = [c[:, i[:numCC]] for c in comp]
    if returncorrs:
        return comp, tcorrs
    else:
        return comp

def _listcorr(a):
    '''Returns pairwise row correlations for all items in array as a list of matrices
    '''
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j>i:
                corrs[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0,1]) for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs

def recon(data, comp, corronly=False, kernelcca = True):
    nT = data[0].shape[0]
    # Get canonical variates and CCs
    if kernelcca:
        ws = _listdot(data, comp)
    else:
        ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return corrs, ws, ccomp

def _listdot(d1, d2): return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]

def _make_kernel(d, normalize=True, ktype="linear", sigma=1.0):
    '''Makes a kernel for data d
      If ktype is "linear", the kernel is a linear inner product
      If ktype is "gaussian", the kernel is a Gaussian kernel with sigma = sigma
    '''
    if ktype == "linear":
        d = np.nan_to_num(d)
        cd = _demean(d)
        kernel = np.dot(cd, cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform
        # this is an NxD matrix, where N is number of items and D its dimensionalites
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / sigma ** 2)
    kernel = (kernel + kernel.T) / 2.
    kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel

def _demean(d): return d-d.mean(0)