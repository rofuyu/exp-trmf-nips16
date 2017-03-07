
#include <time.h>
#include <cstddef>
#include "tron.h"
#include "smat.h"
#include "dmat.h"
#include "dbilinear.h"

#define INF HUGE_VAL

static void print_string_stdout(const char *s) { fputs(s,stdout); fflush(stdout); }
static void print_null(const char *){}
typedef void (*print_fun_ptr)(const char *);
template<typename T>
static print_fun_ptr get_print_fun(T *param) {
//static void (*(get_print_fun(bilinear_param_t *param)))(const char *) {
	if(param->verbose == 0) return print_null;
	else if(param->verbose == 1) return print_null;
	else return print_string_stdout;
}

// AutoRegressive regularization + squared-L2 loss + full observation
class arr_ls_fY_dX: public function { // {{{
	const arr_prob_t *prob;
	const arr_param_t *param;
	smat_t &Y;
	double *X, *lag_val;
	size_t *lag_idx;
	size_t m, T, k, lag_size;
	double lambdaI, lambdaAR;
	double trYTY;
	double *YTX; // Y^T*X
	double *XTX; // X^TX
	double *WTW;
	double *z;
	double *D;

	public:
	int get_nr_variable(void) {return(int) (prob->T*prob->k);}
	arr_ls_fY_dX(const arr_prob_t *prob, const arr_param_t *param):
		prob(prob), param(param),
		Y(*prob->Y), X(prob->X), lag_val(prob->lag_val), lag_idx(prob->lag_idx),
		m(prob->m), T(prob->T), k(prob->k), lag_size(prob->lag_size),
		lambdaI(param->lambdaI), lambdaAR(param->lambdaAR){
		YTX = MALLOC(double, T*k);
		XTX = MALLOC(double, k*k);
		WTW = MALLOC(double, k*k);
		trYTY = do_dot_product(Y.val_t, Y.val_t, Y.nnz);
		//init(); // required init() before solve.
	}
	~arr_ls_fY_dX() {
		if(YTX) free(YTX);
		if(XTX) free(XTX);
		if(WTW) free(WTW);
	}
	void init() {
		smat_t Yt = Y.transpose();
		smat_x_dmat(Yt, X, k, YTX);
		doHTH(X,XTX,m,k);
	}
	double fun(double *w) {
		double f = 0;
		f += 0.5*trYTY;
		doHTH(w, WTW, T, k);
		f += 0.5*do_dot_product(WTW, XTX, k*k);
		f -= do_dot_product(YTX, w, T*k);
		f += 0.5*lambdaI*do_dot_product(w, w, T*k);
		if(lag_size > 0) {
			double AR_val = 0;
			size_t midx = lag_idx[lag_size-1];
#pragma omp parallel for reduction(+:AR_val)
			for(size_t t = midx; t < T; t++) {
				double tmp_val = 0;
				for(size_t s = 0; s < k; s++) {
					double residual = w[k*t+s];
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						residual -= lag_val[k*d+s] * w[k*(t-lag)+s];
					}
					tmp_val += residual*residual;
				}
				AR_val += tmp_val;
			}
			f += 0.5*lambdaAR*AR_val;
		}
		return f;
	}
	void grad(double *w, double *g) {
		do_copy(YTX, g, T*k);
		doVM(1.0, w, XTX, -1.0, g, T, k);
		do_axpy(lambdaI, w, g, T*k);
		if(lag_size > 0) {
			size_t midx = lag_idx[lag_size-1];
#pragma omp parallel for
			for(size_t s = 0; s < k; s++) {
				for(size_t t = midx; t < T; t++) {
					double residual = w[k*t+s];
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						residual -= lag_val[k*d+s] * w[k*(t-lag)+s];
					}
					g[k*t+s] += lambdaAR*residual;
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						g[k*(t-lag)+s] -= lambdaAR*residual*lag_val[k*d+s];
					}
				}
			}
		}
	}
	void Hv(double *v, double *Hv) {
		doVM(1.0, v, XTX, 0.0, Hv, T, k);
		do_axpy(lambdaI, v, Hv, T*k);
		if(lag_size > 0) {
			size_t midx = lag_idx[lag_size-1];
#pragma omp parallel for
			for(size_t s = 0; s < k; s++) {
				for(size_t t = midx; t < T; t++) {
					double residual = v[k*t+s];
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						residual -= lag_val[k*d+s] * v[k*(t-lag)+s];
					}
					Hv[k*t+s] += lambdaAR*residual;
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						Hv[k*(t-lag)+s] -= lambdaAR*residual*lag_val[k*d+s];
					}
				}
			}
		}
	}
}; // }}}

// AutoRegressive regularization + squared-L2 loss + missing-value
class arr_ls_mY_dX: public function { // {{{
	const arr_prob_t *prob;
	const arr_param_t *param;
	smat_t &Y;
	double *X, *lag_val;
	size_t *lag_idx;
	size_t m, T, k, lag_size;
	double lambdaI, lambdaAR;

	public:
	int get_nr_variable(void) {return(int) (prob->T*prob->k);}
	arr_ls_mY_dX(const arr_prob_t *prob, const arr_param_t *param):
		prob(prob), param(param),
		Y(*prob->Y),X(prob->X), lag_val(prob->lag_val), lag_idx(prob->lag_idx),
		m(prob->m), T(prob->T), k(prob->k), lag_size(prob->lag_size),
	   lambdaI(param->lambdaI), lambdaAR(param->lambdaAR) {}
	~arr_ls_mY_dX() {}
	void init() {}
	double fun(double *w) {
		double f = 0;
#pragma omp parallel for schedule(dynamic,32) reduction(+:f)
		for(size_t i = 0; i < Y.rows; i++) {
			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
				size_t j = Y.col_idx[idx];
				double sum = -Y.val_t[idx];
				for(size_t t = 0; t < k; t++)
					sum += X[i*k+t]*w[j*k+t];
				f += sum*sum;
			}
		}
		f *= 0.5;
		f += 0.5*lambdaI*do_dot_product(w, w, T*k);
		if(lag_size > 0) {
			double AR_val = 0;
			size_t midx = lag_idx[lag_size-1];
			for(size_t t = midx; t < T; t++) {
				double tmp_val = 0;
				for(size_t s = 0; s < k; s++) {
					double residual = w[k*t+s];
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						residual -= lag_val[k*d+s] * w[k*(t-lag)+s];
					}
					tmp_val += residual*residual;
				}
				AR_val += tmp_val;
			}
			f += 0.5*lambdaAR*AR_val;
		}
		return f;
	}
	void grad(double *w, double *g) {
#pragma omp parallel for schedule(dynamic,32)
		for(size_t j = 0; j < Y.cols; j++) {
			for(size_t t = 0; t < k; t++)
				g[j*k+t] = 0;
			for(long idx = Y.col_ptr[j]; idx != Y.col_ptr[j+1]; idx++) {
				size_t i = Y.row_idx[idx];
				double sum = -Y.val[idx];
				for(size_t t = 0; t < k; t++)
					sum += X[i*k+t]*w[j*k+t];
				for(size_t t = 0; t < k; t++)
					g[j*k+t] += sum*X[i*k+t];
			}
		}
		do_axpy(lambdaI, w, g, T*k);
		if(lag_size > 0) {
			size_t midx = lag_idx[lag_size-1];
			for(size_t t = midx; t < T; t++) {
				for(size_t s = 0; s < k; s++) {
					double residual = w[k*t+s];
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						residual -= lag_val[k*d+s]*w[k*(t-lag)+s];
					}
					g[k*t+s] += lambdaAR*residual;
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						g[k*(t-lag)+s] -= lambdaAR*residual*lag_val[k*d+s];
					}
				}
			}
		}
	}
	void Hv(double *v, double *Hv) {
#pragma omp parallel for schedule(dynamic,32)
		for(size_t j = 0; j < Y.cols; j++) {
			for(size_t t = 0; t < k; t++)
				Hv[j*k+t] = 0;
			for(long idx = Y.col_ptr[j]; idx != Y.col_ptr[j+1]; idx++) {
				size_t i = Y.row_idx[idx];
				double sum = 0;
				for(size_t t = 0; t < k; t++)
					sum += X[i*k+t]*v[j*k+t];
				for(size_t t = 0; t < k; t++)
					Hv[j*k+t] += sum*X[i*k+t];
			}
		}
		do_axpy(lambdaI, v, Hv, T*k);
		if(lag_size > 0) {
			size_t midx = lag_idx[lag_size-1];
			for(size_t t = midx; t < T; t++) {
				for(size_t s = 0; s < k; s++) {
					double residual = v[k*t+s];
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						residual -= lag_val[k*d+s]*v[k*(t-lag)+s];
					}
					Hv[k*t+s] += lambdaAR*residual;
					for(size_t d = 0; d < lag_size; d++) {
						size_t lag = lag_idx[d];
						Hv[k*(t-lag)+s] -= lambdaAR*residual*lag_val[k*d+s];
					}
				}
			}
		}
	}
}; // }}}
/*
 *  Case with X = I
 *  W = argmin_{W} 0.5*|Y - W*H'|^2 + 0.5*lambda*|W|^2
 *  W = argmin_{W}  C * |Y - W*H'|^2 +  0.5*|W|^2
 *    C = 1/(2*lambda)
 */
struct l2r_ls_fY_IX_chol: public solver_t {// {{{
	smat_t *Y;
	double *H, *HTH, lambda;
	double *YH, *kk_buf, trYTY; // for calculation of fun()
	size_t m, k;
	bool done_init;

	l2r_ls_fY_IX_chol(bilinear_prob_t *prob, bilinear_param_t *param):
		Y(prob->Y), H(prob->H), HTH(NULL), lambda(param->lambda), YH(NULL), kk_buf(NULL),
		m(prob->m), k(prob->k), done_init(false) {
		HTH = MALLOC(double, k*k);
		YH = MALLOC(double, m*k);
		kk_buf = MALLOC(double, k*k);
		trYTY = do_dot_product(Y->val, Y->val, Y->nnz);
	}
	~l2r_ls_fY_IX_chol() { if(HTH) free(HTH); if(YH) free(YH); if(kk_buf) free(kk_buf);}
	void init_prob() {
		doHTH(H, HTH, Y->cols, k);
		for(size_t t= 0; t < k; t++)
			HTH[t*k+t] += lambda;
		smat_x_dmat(*Y, H, k, YH);
		done_init = true;
	}
	void solve(double *W) {
		if(!done_init) {init_prob();}
		do_copy(YH, W, m*k);
		ls_solve_chol_matrix(HTH, W, k, m);
		done_init = false; // ls_solve_chol modifies HTH...
	}
	double fun(double *w) {
		if(!done_init) {init_prob();}
		double obj = 0;
		obj += trYTY;
		doHTH(w, kk_buf, m, k);
		obj += do_dot_product(kk_buf, HTH, k*k);
		obj -= 2.0*do_dot_product(w, YH, m*k);
		return 0.5*obj;
	}
}; // }}}
struct l2r_ls_mY_IX_chol: public solver_t { // {{{
	smat_t *Y;
	double *H, **Hessian_set, lambda;
	size_t m, k, nr_threads;

	l2r_ls_mY_IX_chol(bilinear_prob_t *prob, bilinear_param_t *param):
		Y(prob->Y), H(prob->H), Hessian_set(NULL), lambda(param->lambda),
		m(prob->m), k(prob->k) {//{{{
		nr_threads = omp_get_max_threads();
		Hessian_set = MALLOC(double*, nr_threads);
		for(size_t i = 0; i < nr_threads; i++)
			Hessian_set[i] = MALLOC(double, k*k);
	} // }}}
	~l2r_ls_mY_IX_chol(){ // {{{
		for(size_t i = 0; i < nr_threads; i++)
			if(Hessian_set[i]) free(Hessian_set[i]);
		free(Hessian_set);
	} // }}}
	void init_prob() {}
	void solve(double *W) { // {{{
		smat_t &Y = *(this->Y);
#pragma omp parallel for schedule(dynamic,64)
		for(size_t i = 0; i < Y.rows; ++i) {
			size_t nnz_i = Y.nnz_of_row(i);
			if(!nnz_i) continue;
			int tid = omp_get_thread_num(); // thread ID
			double *Wi = &W[i*k];
			double *Hessian = Hessian_set[tid];
			double *y = Wi;
			memset(Hessian, 0, sizeof(double)*k*k);
			memset(y, 0, sizeof(double)*k);

			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; ++idx) {
				const double *Hj = &H[k*Y.col_idx[idx]];
				for(size_t s = 0; s < k; ++s) {
					y[s] += Y.val_t[idx]*Hj[s];
					for(size_t t = s; t < k; ++t)
						Hessian[s*k+t] += Hj[s]*Hj[t];
				}
			}
			for(size_t s = 0; s < k; ++s) {
				for(size_t t = 0; t < s; ++t)
					Hessian[s*k+t] = Hessian[t*k+s];
				Hessian[s*k+s] += lambda;
			}
			ls_solve_chol_matrix(Hessian, y, k);
		}
	} // }}}
	double fun(double *W) {
		smat_t &Y = *(this->Y);
		double loss = 0;
#pragma omp parallel for reduction(+:loss) schedule(dynamic,32)
		for(size_t i = 0; i < Y.rows; i++) {
			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
				double err = -Y.val_t[idx];
				size_t j = Y.col_idx[idx];
				for(size_t s = 0; s < k; s++)
					err += W[i*k+s]*H[j*k+s];
				loss += err*err;
			}
		}
		double reg = do_dot_product(W,W,m*k);
		return 0.5*(loss + lambda*reg);
	}
}; // }}}

arr_solver::arr_solver(arr_prob_t *prob, arr_param_t *param): prob(prob), param(param), fun_obj(NULL), tron_obj(NULL), solver_obj(NULL), done_init(false) { // {{{
	if(prob->lag_size > 0) {
		switch(param->solver_type) {
			case ARR_LS_FULL:
				fun_obj = new arr_ls_fY_dX(prob, param);
				break;
			case ARR_LS_MISSING:
				fun_obj = new arr_ls_mY_dX(prob, param);
				break;
			default:
				fprintf(stderr, "Solver not supported\n");
				break;
		}
		fflush(stdout);
		int max_cg_iter = param->max_cg_iter;
		if(max_cg_iter >= fun_obj->get_nr_variable())
			max_cg_iter = fun_obj->get_nr_variable();
	//	printf("max_cg_iter %d\n", max_cg_iter);
		bool pure_cg = true;
		tron_obj = new TRON(fun_obj, param->eps, param->max_tron_iter, max_cg_iter, pure_cg, param->eps);
		tron_obj->set_print_string(get_print_fun(param));
	} else { // fall back to the simple matrix factorization
		Yt = prob->Y->transpose();
		tmp_prob = bilinear_prob_t(&Yt, NULL, prob->X, prob->T, prob->k, bilinear_prob_t::Identity);
		switch(param->solver_type) {
			case ARR_LS_FULL:
				solver_obj = new l2r_ls_fY_IX_chol(&tmp_prob, param);
				break;
			case ARR_LS_MISSING:
				solver_obj = new l2r_ls_mY_IX_chol(&tmp_prob, param);
				break;
			default:
				fprintf(stderr, "Solver not supported\n");
				break;
		}
	}
} //}}}

double cal_rmse(smat_t &testY, double *W, double *H, size_t k) { // {{{
	double rmse = 0.0;
	for(size_t i = 0; i < testY.rows; i++) {
		for(long idx = testY.row_ptr[i]; idx != testY.row_ptr[i+1]; idx++) {
			size_t j = testY.col_idx[idx];
			double true_val = testY.val_t[idx], pred_val = 0.0;
			for(size_t t = 0; t < k; t++)
				pred_val += W[k*i+t]*H[k*j+t];
			rmse += (pred_val-true_val)*(pred_val-true_val);
		}
	}
	return sqrt(rmse/(double)testY.nnz);
} // }}}


