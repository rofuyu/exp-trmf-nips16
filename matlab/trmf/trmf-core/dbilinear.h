#ifndef _GRAPH_REG_H
#define _GRAPH_REG_H


#include "smat.h"
#include "tron.h"
#include <algorithm>

// Solvers
enum {
	  L2R_LS_FULL=0,  L2R_LS_MISSING=1,
	  O2R_LS_FULL=10, O2R_LS_MISSING=11,
	  GLR_LS_FULL=20, GLR_LS_MISSING=21,
	  ARR_LS_FULL=30, ARR_LS_MISSING=31,
};

// Bilinear Problems with L2/output Regularization Problems
// O2R: min_W sum_ij loss(Y_ij, x_i^T W h_j) + 0.5 * sum_i lambda |W^T x_i)|^2
// L2R: min_W sum_ij loss(Y_ij, x_i^T W h_j) + 0.5 * lambda |W|^2
struct bilinear_prob_t { // {{{
	smat_t *Y; // m*l sparse matrix
	union {
		double *X; // m*f array row major X(i,s) = X[k*i+s]
		smat_t *spX; // m*f sparse matrix
	};
	double *H; // l*k array row major H(j,s) = H[k*i+s]
	double *W; // f*k array row major W(t,s) = W[k*t+s]
	size_t m;  // #instance
	size_t f;  // #features
	size_t l;  // #labels
	size_t k;  // row-rank dimension
	int X_type;// type of X: either 0 for dense, 1 for sparse, 2 for Identity
	enum {Dense=0, Sparse=1, Identity=2};
	bilinear_prob_t(){}
	bilinear_prob_t(smat_t *Y, void *X, double *H, size_t f, size_t k, int X_type=Dense, double *W=NULL) {
		this->Y = Y;
		if(X_type == Dense || X_type == Identity)
			this->X = (double*) X;
		else
			this->spX = (smat_t*) X;
		this->k = k;
		this->m = Y->rows;
		this->f = f;
		this->l = Y->cols;
		this->W = W;
		this->H = H;
		this->X_type = X_type;
	}
};

struct bilinear_param_t {
	int solver_type;
	double lambda;
	int max_tron_iter, max_cg_iter;
	double eps;
	int threads;
	int verbose;
	int weighted_reg;
	int use_chol;
	bilinear_param_t() {
		solver_type = O2R_LS_FULL;
		lambda = 0.1;
		max_tron_iter = 3;
		max_cg_iter = 20;
		eps = 0.1;
		verbose = 0;
		threads = 4;
		weighted_reg = 0;
		use_chol = 0;
	}
}; // }}}

// AutoRegressive Regularized Problems
// min_W sum_ij 0.5*|Y - XW^T|^2 + 0.5*lambda1 sum_{r=s}^k sum_{t= midx+1}^T G_{sl} |w_{ts} - w_{(t-l)s}|^2 + 0.5*lambda2 sum_{t=1}^T |w_t|^2
class arr_prob_t { // {{{
	public:
		smat_t *Y; // n*T sparse matrix
		double *X; // n*k array row major X(i,s) = X[k*i+s]
		double *W; // T*k array row major W(t,s) = W[k*t+s]
		size_t m; // #instances
		size_t T; // #time stamps
		size_t k; // low-rank dimension
		size_t *lag_idx; // lag index
		size_t lag_size;  // size of lag set
		double *lag_val; // lag_size*k array for lag valus lag_val(l, s) = lag_val[k*l + s]
		arr_prob_t(smat_t *Y, double *X, size_t k, size_t lag_size, size_t *lag_idx, double *lag_val, double *W=NULL) {
			this->Y = Y;
			this->X = X;
			this->k = k;
			this->m = Y->rows;
			this->T = Y->cols;
			this->lag_size = lag_size;
			this->lag_idx = lag_idx;
			this->lag_val = lag_val;
			this->W = W;
		}
};

class arr_param_t : public bilinear_param_t{
	public:
		double lambdaAR;
		double lambdaI;
		arr_param_t(): bilinear_param_t() {
			solver_type = ARR_LS_FULL;
			max_tron_iter = 3;
			max_cg_iter = 20;
			lambdaAR = 0.1;
			lambdaI = lambda; //0.1;
			eps = 0.1;
			verbose = 0;
			threads = 4;
		}
};// }}}


struct solver_t {
	virtual void init_prob() = 0;
	virtual void solve(double *w) = 0;
	virtual double fun(double *w) {return 0;}
	virtual void grad(double *w, double *g) {return;}
	virtual void Hv(double *s, double *Hs) {return;}
	virtual ~solver_t(){}
};
struct arr_solver: public solver_t { // {{{
	arr_prob_t *prob;
	arr_param_t *param;
	function *fun_obj;
	TRON *tron_obj;
	solver_t *solver_obj;
	bool done_init;
	smat_t Yt;
	bilinear_prob_t tmp_prob;

	arr_solver(arr_prob_t *prob, arr_param_t *param);
	~arr_solver() {
		if(tron_obj) delete tron_obj;
		if(fun_obj) delete fun_obj;
		if(solver_obj) delete solver_obj;
	}

	void init_prob() {
		if(fun_obj) fun_obj->init();
		else if(solver_obj) solver_obj->init_prob();
		done_init = true;
	}
	void set_eps(double eps) {tron_obj->set_eps(eps);}
	void solve(double *w) {
		if(!done_init) {init_prob();}
		if(tron_obj) {
			tron_obj->tron(w, false);
			//tron_obj->tron(w, true); // zero init for w
		} else if(solver_obj) {
			solver_obj->solve(w);
		}
	}
	double fun(double *w) {
		if(!done_init) {init_prob();}
		if(fun_obj)
			return fun_obj->fun(w);
		else if(solver_obj)
			return solver_obj->fun(w);
		else
			return 0;
	}
	void grad(double *w, double *g) {
		if(!done_init) {init_prob();}
		if(fun_obj)  {
			fun_obj->grad(w, g);
		} else
			return ;
	}
	void Hv(double *s, double *Hs) {
		if(!done_init) {init_prob();}
		if(fun_obj)
			fun_obj->Hv(s, Hs);
	}
}; // }}}

#ifdef __cplusplus
extern "C" {
#endif

double cal_rmse(smat_t &testY, double *W, double *H, size_t k);
#ifdef __cplusplus
}
#endif

#endif /* _GRAPH_REG_H */
