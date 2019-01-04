#ifndef _TRMF_H
#define _TRMF_H

#include "rf_tron.h"
#include "rf_matrix.h"

#ifndef ValueType
#define ValueType float
#endif

typedef size_t IndexType;

typedef general_matrix_wrapper<ValueType> gmat_wrapper_t;
typedef gmat_wrapper_t::gvec_t gvec_t;
typedef gmat_wrapper_t::dvec_t dvec_t;
typedef gmat_wrapper_t::svec_t svec_t;
typedef gmat_wrapper_t::gmat_t gmat_t;
typedef gmat_wrapper_t::dmat_t dmat_t;
typedef gmat_wrapper_t::smat_t smat_t;
//typedef std::vector<IndexType> ivec_t;
typedef dense_vector<uint32_t> ivec_t;


// Solvers
enum {
    ARR_LS_FULL=30,
    ARR_LS_MISSING=31,
};


// AutoRegressive Regularized Problems
// min_{W}    0.5 * || Y - W H^T ||^2  .... loss
//          + 0.5 * lambdaAR * T_AR(W | LagSet, Theta) .... temporal regularization over W
//          + 0.5 * lambdaI * || W ||_F^2
//    where
//      Y        = m * n time series (each column is a time series of length m, i.e., m timestamps)
//      H        = n * k factors (each row is a k-dimensional vector for the jth time series)
//      LegSet   = lag_set (an integer array to denote the lag indices to look back)
//      Theta    = |L| * k factors (each column is a vector of AR coefficients for t-th latent dimension w.r.t the lag set L)
//      lambdaAR = regularization parameter for T_AR(W | LagSet, \Theta)
//      lambdaI  = regularization parameter for ||W||_F^2
//      T_AR(W | LagSet, Theta) =
//         0.5 * \sum_{i = max(LagSet) + 1}^{m} {
//                   || \bw_{i} - \sum_{lag \in L} \bw_{i - lag} \diag(\btheta_{lag}) ||^2
//               }
//      =  0.5 * \sum_{i = max(LagSet) + 1}^{m} {
//                    \sum_{t = 1}^{k} {
//                        || W_{i, t}
//                           - sum_{s = 1}^{|LagSet|} { Theta_{s, t} * W_{i - L(s), t} ||^2
//                    }
//               }
//

struct arr_prob_t { // {{{
    gmat_t *Y;       // m*n sparse matrix
    dmat_t *H;       // n*k array row major H(j,t) = H[k*j+t]
    size_t m;        // #time stamps = Y.rows
    size_t n;        // #time series = Y.cols
    size_t k;        // #low-rank dimension
    ivec_t *lag_set; // lag set L (sorted is needed)
    dmat_t *lag_val; // |L| * k , i.e. Theta  COLMAJOR

    arr_prob_t(gmat_t *Y, dmat_t *H, ivec_t *lag_set=NULL, dmat_t *lag_val=NULL):
        Y(Y), H(H), m(Y->rows), n(Y->cols), k(H->cols),
        lag_set(lag_set), lag_val(lag_val) {
        if(lag_set != NULL) {
            //std::sort(lag_set->begin(), lag_set->end());
            assert(lag_set->size() == lag_val->rows);
            assert(lag_val->cols == H->cols);
        }
    }
};

struct arr_param_t {
    int solver_type;
    double lambdaAR;
    double lambdaI;
    double eps;        // eps for TRON
    double eps_cg;     // eps used for CG
    int max_tron_iter;
    int max_cg_iter;
    int threads;
    int verbose;

    arr_param_t() {
        solver_type = ARR_LS_FULL;
        //solver_type = ARR_LS_MISSING;
        lambdaI = 0.1;
        lambdaAR = 0.1;
        eps = 0.1;
        eps_cg = 0.1;
        max_tron_iter = 2;
        max_cg_iter = 10;
        verbose = 1;
        threads = 4;
    }
};

struct trmf_prob_t {
    gmat_t *Y;       // m*n sparse matrix
    gmat_t *Yt;      // Y transpose
    ivec_t *lag_set; // lag set L (sorted is needed)
    size_t m;        // #time stamps = Y.rows
    size_t n;        // #time series = Y.cols
    size_t k;        // #low-rank dimension
    trmf_prob_t(gmat_t *Y, gmat_t *Yt, ivec_t *lag_set, size_t k):
        Y(Y), Yt(Yt), lag_set(lag_set), m(Y->rows), n(Y->cols), k(k) {}
};

struct trmf_param_t : public arr_param_t {
    double lambdaLag;
    int max_iter;
    int period_W;
    int period_H;
    int period_Lag;

    trmf_param_t(): arr_param_t() {
        lambdaLag = 0.1;
        max_iter = 10;
        period_W = 1;
        period_H = 1;
        period_Lag = 2;
    }
};

// }}}


struct solver_t {
    virtual void init_prob() = 0;
    virtual void solve(void *w) = 0;
    virtual double fun(void *w) { return 0; }
    virtual ~solver_t(){}
};

template<typename val_type>
struct arr_solver: public solver_t { // {{{
    arr_prob_t *prob;
    arr_param_t *param;
    function *fun_obj;
    TRON<val_type> *tron_obj;
    solver_t *solver_obj;
    bool done_init;

    arr_solver(arr_prob_t *prob, arr_param_t *param);
    arr_solver(const arr_solver& other) {}

    void zero_init() {
        prob = NULL;
        param = NULL;
        fun_obj = NULL;
        tron_obj = NULL;
        solver_obj = NULL;
        done_init = false;
    }

    ~arr_solver() {
        if(tron_obj) { delete tron_obj; }
        if(fun_obj) { delete fun_obj; }
        if(solver_obj) { delete solver_obj; }
        zero_init();
    }

    void init_prob() {
        if(fun_obj) {
            fun_obj->init();
        } else if(solver_obj) {
            solver_obj->init_prob();
        }
        done_init = true;
    }

    void set_eps(double eps) { tron_obj->set_eps(eps); }

    void solve(void *w) {
        if(!done_init) {
            this->init_prob();
        }
        if(tron_obj) {
            bool set_w_to_zero = false;
            tron_obj->tron((val_type*)w, set_w_to_zero);
            //tron_obj->tron(w, true); // zero init for w
        } else if(solver_obj) {
            solver_obj->solve(w);
        }
    }

    double fun(void *w) {
        if(!done_init) {
            init_prob();
        }
        if(fun_obj) {
            return fun_obj->fun(w);
        } else if(solver_obj) {
            return solver_obj->fun(w);
        } else {
            return 0;
        }
    }
}; // }}}


extern "C" {

void c_trmf_train(const PyMatrix *pyY, uint32_t *py_lag_set, uint32_t py_lag_size,
        PyMatrix *pyW, PyMatrix *pyH, PyMatrix *pylag_val, int warm_start,
        double lambdaI, double lambdaAR, double lambdaLag,
        int32_t max_iter, int32_t period_W, int32_t period_H, int32_t period_Lag,
        int32_t threads, int32_t missing, int32_t verbose);
};

#endif /* _TRMF_H */
