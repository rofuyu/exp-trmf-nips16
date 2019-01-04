#!/usr/bin/env python

import scipy as sp
import trmf

Y = sp.load('datasets/electricity.npy')
#Y = Y[:-(7 * 24), :]

lag_set = sp.array(list(range(1, 25)) + list(range(7 * 24, 8 * 24)), dtype=sp.uint32)
#lag_set = sp.array(list(range(1, 7 * 24 + 1)), dtype=sp.uint32)
k = 60
lambdaI = 0.5
lambdaAR = 125
lambdaLag = 2
window_size = 24
nr_windows = 7
max_iter = 40
threads=40
seed=0
missing = False
transform = True
threshold=None

metrics = trmf.rolling_validate(Y, lag_set, k, window_size, nr_windows, lambdaI, lambdaAR, lambdaLag,
        max_iter=max_iter, threshold=threshold, transform=transform, threads=threads, seed=seed, missing=missing)
print(metrics)



