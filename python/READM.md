This is a README for an Python experimental code of the paper 

H.-F. Yu et al. Temporal Regularized Matrix Factorization for High Dimensional Time Series Prediction. NIPS 2016


Requirements
============
    * Gnu GCC 
    * python3.4+
	* scipy
	* numpy
	* mkl

Install
=======
    > python3 -m venv trmf-env
	> source trmf-env/bin/activate
    > (trmf-env) cd exp-trmf-nips16/python
	> (trmf-env) pip install -r requirements.txt
	> (trmf-env) pip install .

Dataset
=======
    If you have git client with lfs support, the two datasets (traffic and electrcity) should already downloaded in 
    `exp-trmf-nips16/python/exp-scripts/datasets/`. Otherwise, you can download it manually by

	> wget https://github.com/rofuyu/exp-trmf-nips16/raw/master/python/exp-scripts/datasets/electricity.npy -O exp-trmf-nips16/python/exp-scripts/datasets/electricity.npy
	> wget https://github.com/rofuyu/exp-trmf-nips16/raw/master/python/exp-scripts/datasets/traffic.npy -O exp-trmf-nips16/python/exp-scripts/datasets/traffic.npy

	The above two datasets are essentially a numpy array with the shape = number of time stamps * number of time series stored in a row major format. 

Usage
=====
	> (trmf-env) cd exp-scripts 
	> (trmf-env) python run_traffic.py
	> (trmf-env) python run_electricity.py
