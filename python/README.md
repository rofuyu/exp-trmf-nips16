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

Install for MacOS
=================
    Make sure you have Gnu GCC available in your Mac (you can get it by `brew install gcc`). Assume the GCC version is 8.
    > python3 -m venv trmf-env
	> source trmf-env/bin/activate
    > (trmf-env) cd exp-trmf-nips16/python
	> (trmf-env) pip install -r requirements.txt
	> (trmf-env) CC=gcc-8 CXx=g++-8 pip install .

Dataset
=======
    You can download it manually by

	> cd exp-scripts/datasets; ./download-data.sh; cd ../..

	The above two datasets are essentially a numpy array with the shape = (number of time stamps, number of time series) stored in a row major format. 

Usage
=====
	> (trmf-env) cd exp-scripts 
	> (trmf-env) python run_traffic.py
	> (trmf-env) python run_electricity.py
