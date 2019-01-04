#!/usr/bin/env python

import scipy as sp
import trmf

Y = sp.load('datasets/traffic.npy')
#Y = Y.astype(sp.float32)
lag_set = sp.array(list(range(1, 25)) + list(range(7 * 24, 8 * 24)), dtype=sp.uint32)
k = 40
lambdaI = 2
lambdaAR = 625
lambdaLag = 0.5
window_size = 24
nr_windows = 7
max_iter = 40
threshold = 0
threads=40
seed=0
missing = False
transform = True
threshold=None

metrics = trmf.rolling_validate(Y, lag_set, k, window_size, nr_windows, lambdaI, lambdaAR, lambdaLag,
        max_iter=max_iter, threshold=threshold, transform=transform, threads=threads, seed=seed, missing=missing)
print(metrics)

