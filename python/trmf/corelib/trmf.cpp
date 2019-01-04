
#include <time.h>
#include <cstddef>
#include "trmf.h"

// print function {{{
static void print_string_stdout(const char *s) { fputs(s,stdout); fflush(stdout); }
static void print_null(const char *){}
typedef void (*print_fun_ptr)(const char *);
template<typename T>
static print_fun_ptr get_print_fun(T *param) {
    if(param->verbose == 0) return print_null;
    else if(param->verbose == 1) return print_null;
    else return print_string_stdout;
}
// }}}
//
// wallclock function
struct walltime_clock_t {
    int64_t last_time;
    walltime_clock_t(): last_time(0) {}

    int64_t tic() { return last_time = now(); }
    double toc() { return static_cast<double>(now() - last_time) / 1e6; }
    int64_t now() {
        struct timespec tw;
        clock_gettime(CLOCK_MONOTONIC, &tw);
        return tw.tv_sec * (1000000000L) + tw.tv_nsec;
    }
};

// AutoRegressive Regularized Function
template<typename val_type>
class arr_base_IX: public function {
    public:
        typedef arr_prob_t prob_t;
        typedef arr_param_t param_t;
        typedef general_matrix_wrapper<val_type> gmat_wrapper_t;
        typedef typename gmat_wrapper_t::gmat_t gmat_t;
        typedef typename gmat_wrapper_t::dmat_t dmat_t;
        typedef typename gmat_wrapper_t::smat_t smat_t;
        typedef typename gmat_wrapper_t::eye_t eye_t;
        typedef typename gmat_wrapper_t::gvec_t gvec_t;
        typedef typename gmat_wrapper_t::dvec_t dvec_t;
        typedef typename gmat_wrapper_t::svec_t svec_t;;

    protected:
        const prob_t *prob;
        const param_t *param;

        // const reference to prob
        const gmat_t &Y;
        const dmat_t &H;
        const size_t &m;
        const size_t &n;
        const size_t &k;
        const double &lambdaI;
        const double &lambdaAR;

    public:
        arr_base_IX(const prob_t *prob, const param_t *param):
            prob(prob), param(param), Y(*(prob->Y)), H(*(prob->H)),
            m(prob->m), n(prob->n), k(prob->k),
            lambdaI(param->lambdaI), lambdaAR(param->lambdaAR) { }

        int get_nr_variable(void) { return static_cast<int>(prob->m * prob->k); }

        virtual void init() {}

        double fun(void *w) {
            const dmat_t W(m, k, ROWMAJOR, (val_type*) w); // view constructor
            double f = 0;
            if(lambdaI > 0) {
                f += 0.5 * lambdaI * do_dot_product(W, W);
            }
            if(prob->lag_set != NULL && lambdaAR > 0) {
                const ivec_t& lag_set = *(prob->lag_set);
                const dmat_t& lag_val = *(prob->lag_val);
                size_t midx = lag_set.back(); // supposed to be the max index in lag_set
                double AR_val = 0;
#pragma omp parallel for reduction(+:AR_val)
                for(size_t i = midx; i < m; i++) {
                    double tmp_val = 0;
                    for(size_t t = 0; t < k; t++) {
                        double residual = W.at(i, t);
                        for(size_t l = 0; l < lag_set.size(); l++) {
                            size_t lag = lag_set[l];
                            residual -= lag_val.at(l, t) * W.at(i - lag, t);
                        }
                        tmp_val += residual * residual;
                    }
                    AR_val += tmp_val;
                }
                f += 0.5 * lambdaAR * AR_val;
            }
            return f;
        }

        void grad(void *w, void *g) {
            const dmat_t W(m, k, ROWMAJOR, (val_type*) w); // view constructor
            dmat_t G(m, k, ROWMAJOR, (val_type*) g);       // view constructor
            G.assign(lambdaI, W);        // G <= lambdaI * W
            if(prob->lag_set != NULL && lambdaAR > 0) {
                const ivec_t& lag_set = *(prob->lag_set);
                const dmat_t& lag_val = *(prob->lag_val);
                size_t midx = lag_set.back(); // supposed to be the max index in lag_set
#pragma omp parallel for
                for(size_t t = 0; t < k; t++) {
                    for(size_t i = midx; i < m; i++) {
                        double residual = W.at(i, t);
                        for(size_t l = 0; l < lag_set.size(); l++) {
                            size_t lag = lag_set[l];
                            residual -= lag_val.at(l, t) * W.at(i - lag, t);
                        }
                        G.at(i, t) += lambdaAR * residual;
                        for(size_t l = 0; l < lag_set.size(); l++) {
                            size_t lag = lag_set[l];
                            G.at(i - lag, t) -= lambdaAR * residual * lag_val.at(l, t);
                        }
                    }
                }
            }
        }

        void Hv(void* s, void *Hs) {
            const dmat_t S(m, k, ROWMAJOR, (val_type*) s);   // view constructor
            dmat_t HS(m, k, ROWMAJOR, (val_type*) Hs);       // view constructor
            HS.assign(lambdaI, S);               // HS <= lambdaI * S
            if(prob->lag_set != NULL && lambdaAR > 0) {
                const ivec_t& lag_set = *(prob->lag_set);
                const dmat_t& lag_val = *(prob->lag_val);
                size_t midx = lag_set.back();
#pragma omp parallel for
                for(size_t t = 0; t < k; t++) {
                    for(size_t i = midx; i < m; i++) {
                        double residual = S.at(i, t);
                        for(size_t l = 0; l < lag_set.size(); l++) {
                            size_t lag = lag_set[l];
                            residual -= lag_val.at(l, t) * S.at(i - lag, t);
                        }
                        HS.at(i, t) += lambdaAR * residual;
                        for(size_t l = 0; l < lag_set.size(); l++) {
                            size_t lag = lag_set[l];
                            HS.at(i - lag, t) -= lambdaAR * residual * lag_val.at(l, t);
                        }
                    }
                }
            }
        }

};

// AutoRegressive regularization + squared-L2 loss + full observation
// See arr_prob_t for the mathmatical definitions
template<typename val_type>
class arr_ls_fY_IX: public arr_base_IX<val_type> { // {{{
    public:
        typedef arr_base_IX<val_type> base_t;
        typedef typename base_t::prob_t prob_t;
        typedef typename base_t::param_t param_t;
        typedef typename base_t::gmat_t gmat_t;
        typedef typename base_t::dmat_t dmat_t;
        typedef typename base_t::smat_t smat_t;
        typedef typename base_t::eye_t eye_t;
        typedef typename base_t::gvec_t gvec_t;
        typedef typename base_t::dvec_t dvec_t;
        typedef typename base_t::svec_t svec_t;

    protected:
        double trYTY;
        dmat_t YH;  // m * k => Y * H
        dmat_t HTH; // k * k => H^T * H
        dmat_t WTW; // k * k => W^T * W

    public:
        arr_ls_fY_IX(const prob_t *prob, const param_t *param): base_t(prob, param) {
            trYTY = do_dot_product(this->Y, this->Y);
            YH =  dmat_t(this->m, this->k, ROWMAJOR);
            HTH = dmat_t(this->k, this->k, ROWMAJOR);
            WTW = dmat_t(this->k, this->k, ROWMAJOR);
        }

        void init() {
            trYTY = do_dot_product(this->Y, this->Y);
            gmat_x_dmat(this->Y, this->H, this->YH);
            dmat_x_dmat(this->H.transpose(), this->H, this->HTH);
        }

        double fun(void *w) {
            const dmat_t W(this->m, this->k, ROWMAJOR, (val_type*) w); // view constructor
            double f = base_t::fun(w);
            f += 0.5 * trYTY;
            dmat_x_dmat(W.transpose(), W, WTW);
            f += 0.5 * do_dot_product(WTW, HTH);
            f -= do_dot_product(YH, W);
            return f;
        }

        void grad(void *w, void *g) {
            base_t::grad(w, g);
            const dmat_t W(this->m, this->k, ROWMAJOR, (val_type*) w); // view constructor
            dmat_t G(this->m, this->k, ROWMAJOR, (val_type*) g);       // view constructor

            // G += -YH + WHTH
            do_axpy(-1.0, YH, G);
            dmat_x_dmat(1.0, W, HTH, 1.0, G, G);
        }

        void Hv(void *s, void *Hs) {
            base_t::Hv(s, Hs);
            const dmat_t S(this->m, this->k, ROWMAJOR, (val_type*) s);   // view constructor
            dmat_t HS(this->m, this->k, ROWMAJOR, (val_type*) Hs);       // view constructor
            dmat_x_dmat(1.0, S, HTH, 1.0, HS, HS);
        }
};

// AutoRegressive regularization + squared-L2 loss + partial observation
// Only sparse matrix is allowed for Y in this function
// See arr_prob_t for the mathmatical definitions
template<typename val_type>
class arr_ls_pY_IX: public arr_base_IX<val_type> {
    protected:
        const smat_t& sY;
    public:
        typedef arr_base_IX<val_type> base_t;
        typedef typename base_t::prob_t prob_t;
        typedef typename base_t::param_t param_t;

        arr_ls_pY_IX(const prob_t *prob, const param_t *param): base_t(prob, param), sY(this->Y.get_sparse()) { }

        double fun(void *w) {
            const dmat_t W(this->m, this->k, ROWMAJOR, (val_type*) w); // view constructor
            double f = 0;
#pragma omp parallel for schedule(dynamic,32) reduction(+:f)
            for(size_t i = 0; i < sY.rows; i++) {
                for(size_t idx = sY.row_ptr[i]; idx != sY.row_ptr[i + 1]; idx++) {
                    size_t j = sY.col_idx[idx];
                    double residual = sY.val_t[idx] - do_dot_product(W.get_row(i), this->H.get_row(j));
                    f += residual * residual;
                }
            }
            f *= 0.5;
            f += base_t::fun(w);
            return f;
        }

        void grad(void *w, void *g) {
            base_t::grad(w, g);
            const dmat_t W(this->m, this->k, ROWMAJOR, (val_type*) w); // view constructor
            dmat_t G(this->m, this->k, ROWMAJOR, (val_type*) g);       // view constructor

#pragma omp parallel for schedule(dynamic, 32)
            for(size_t i = 0; i < this->m; i++) {
                for(size_t idx = sY.row_ptr[i]; idx != sY.row_ptr[i + 1]; idx++) {
                    size_t j = sY.col_idx[idx];
                    // residule = <\bw_i, \bh_j> - Y_{ij}
                    // \bg_i = \bg_i + residual * \bh_j
                    double residual = -sY.val_t[idx];
                    for(size_t t = 0; t < this->k; t++) {
                        residual += W.at(i, t) * this->H.at(j, t);
                    }
                    for(size_t t = 0; t < this->k; t++) {
                        G.at(i, t) += residual * this->H.at(j, t);
                    }
                }
            }
        }

        void Hv(void *s, void *Hs) {
            base_t::Hv(s, Hs);
            const dmat_t S(this->m, this->k, ROWMAJOR, (val_type*) s);   // view constructor
            dmat_t HS(this->m, this->k, ROWMAJOR, (val_type*) Hs);       // view constructor
#pragma omp parallel for schedule(dynamic, 32)
            for(size_t i = 0; i < this->m; i++) {
                for(size_t idx = sY.row_ptr[i]; idx != sY.row_ptr[i + 1]; idx++) {
                    size_t j = sY.col_idx[idx];
                    // residule = <\bs_i, \bh_j>
                    // \bhs_i = \bhs_i + residual * \bh_j
                    double residual = 0;
                    for(size_t t = 0; t < this->k; t++) {
                        residual += S.at(i, t) * this->H.at(j, t);
                    }
                    for(size_t t = 0; t < this->k; t++) {
                        HS.at(i, t) += residual * this->H.at(j, t);
                    }
                }
            }
        }
};

/*
 *  Case with X = I
 *  W = argmin_{W} 0.5*|Y-WH'|^2 + 0.5*lambda*|W|^2
 *
 *  W = argmin_{W}  C * |Y - W*H'|^2 +  0.5*|W|^2
 *             where C = 1/(2*lambda)
 * */

template<typename val_type>
struct l2r_ls_fY_IX_chol : public solver_t { // {{{
    const gmat_t &Y;
    const dmat_t &H;
    dmat_t YH;
    dmat_t HTH;
    dmat_t kk_buf;
    double trYTY;
    val_type lambda;
    const size_t &m, &k;
    bool done_init;

    l2r_ls_fY_IX_chol(const gmat_t &Y, const dmat_t &H, val_type lambda):
        Y(Y), H(H), trYTY(0), lambda(lambda), m(Y.rows), k(H.cols), done_init(false) {
        YH = dmat_t(m, k, ROWMAJOR);
        HTH = dmat_t(k, k, ROWMAJOR);
        kk_buf = dmat_t(k, k, ROWMAJOR);
        trYTY = do_dot_product(Y, Y);
    }

    void init_prob() {
        gmat_x_dmat(Y, H, YH);
        dmat_x_dmat(H.transpose(), H, HTH);
        for(size_t t= 0; t < k; t++) {
            HTH.at(t, t) += lambda;
        }
        done_init = true;
    }

    void solve(void *w) {
        if(!done_init) {
            init_prob();
        }
        do_copy(YH.data(), (val_type*) w, m * k);
        ls_solve_chol_matrix_colmajor(HTH.data(), (val_type*) w, k, m);
        // HTH will be changed to a cholesky factorization after the above call.
        // Thus we need to flip done_init to false again
        done_init = false;
    }

    double fun(void *w) {
        if(!done_init) {
            init_prob();
        }
        const dmat_t W(m, k, ROWMAJOR, (val_type*) w);
        double obj = trYTY;
        dmat_x_dmat(W.transpose(), W, kk_buf);
        obj += do_dot_product(kk_buf, HTH);
        obj -= 2.0 * do_dot_product(W, YH);
        obj *= 0.5;
        return obj;
    }
}; // }}}

template<typename val_type>
struct l2r_ls_pY_IX_chol : public solver_t { // {{{
    const gmat_t &Y;
    const dmat_t &H;
    std::vector<dvec_t> Hessian_set;
    val_type lambda;
    const size_t &m, &k;
    size_t nr_threads;

    l2r_ls_pY_IX_chol(const gmat_t& Y, const dmat_t& H, val_type lambda): Y(Y), H(H), Hessian_set(), lambda(lambda), m(Y.rows), k(H.cols) { // {{{
        nr_threads = omp_get_max_threads();
        Hessian_set.resize(nr_threads, dvec_t(k*k));
    } // }}}

    void init_prob() {}

    void solve(void *w) { // {{{
        const smat_t& sY = Y.get_sparse();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t i = 0; i < sY.rows; i++) {
            size_t nnz_i = sY.nnz_of_row(i);
            if(nnz_i == 0) continue;
            int tid = omp_get_thread_num(); // thread ID
            val_type *Wi = &((val_type*)w)[i*k];
            val_type *Hessian = Hessian_set[tid].data();

            val_type *y = Wi;
            memset(Hessian, 0, sizeof(val_type)*k*k);
            memset(y, 0, sizeof(val_type)*k);
            for(size_t idx = sY.row_ptr[i]; idx != sY.row_ptr[i+1]; idx++) {
                const val_type *Hj = H.get_row(sY.col_idx[idx]).data();
                for(size_t s = 0; s < k; s++){
                    y[s] += sY.val_t[idx]*Hj[s];
                    for(size_t t = s; t < k; t++)
                        Hessian[s*k+t] += Hj[s]*Hj[t];
                }
            }
            for(size_t s = 0; s < k; s++) {
                for(size_t t = 0; t < s; t++)
                    Hessian[s*k+t] = Hessian[t*k+s];
                Hessian[s*k+s] += lambda;
            }
            ls_solve_chol(Hessian, y, k);
        }
    } // }}}

    double fun(void *w) { // {{{
        const smat_t& sY = Y.get_sparse();
        double loss = 0;
#pragma omp parallel for reduction(+:loss) schedule(dynamic,32)
        for(size_t i = 0; i < sY.rows; i++) {
            val_type* Wi = &((val_type*)w)[i*k];
            for(size_t idx = sY.row_ptr[i]; idx != sY.row_ptr[i+1]; idx++) {
                double err = -sY.val_t[idx];
                const val_type* Hj = H.get_row(sY.col_idx[idx]).data();
                for(size_t s = 0; s < k; s++)
                    err += Wi[s]*Hj[s];
                loss += err*err;
            }
        }
        double reg = do_dot_product((val_type*) w, (val_type*) w, m*k);
        double obj = 0.5 * (loss + lambda*reg);
        return obj;
    } // }}}
}; // }}}

/*
 * 0.5 * auto-reg(Y, L, Theta)
 * Theta_t = \argmin_{
 *
 * */

template<typename val_type>
struct l2r_autoregressive_solver {
    const dmat_t& T;         // m \times k
    const ivec_t& lag_set;
    const size_t m;
    const size_t k;
    double lambda;
    size_t nr_threads;
    std::vector<dvec_t> univate_series_set;
    std::vector<dmat_t> Hessian_set;
    bool done_init;

    l2r_autoregressive_solver(const dmat_t &T, const ivec_t &lag_set, val_type lambda):
        T(T), lag_set(lag_set), m(T.rows), k(T.cols), lambda(lambda) {
        nr_threads = omp_get_max_threads();
        univate_series_set.resize(nr_threads, dvec_t(m));
        Hessian_set.resize(nr_threads, dmat_t(lag_set.size(), lag_set.size()));
    }

    void init_prob() {
    }

    double lagged_inner_product(const dvec_t& univate_series, size_t start, size_t end, size_t lag_1, size_t lag_2) {
        double ret = 0;
        for(size_t i = start; i < end; i++) {
            ret += univate_series.at(i - lag_1) * univate_series.at(i - lag_2);
        }
        return ret;
    }

    void solve(dmat_t& lag_val) {
        size_t start = lag_set.back();
        size_t end = m;
#pragma omp parallel for schedule(static)
        for(size_t t = 0; t < k; t++) {
            int tid = omp_get_thread_num(); // thread ID
            dvec_t& univate_series = univate_series_set[tid];
            for(size_t i = 0; i < m; i++) {
                univate_series.at(i) = T.at(i, t);
            }
            dvec_t y = lag_val.get_col(t);
            dmat_t& Hessian = Hessian_set[tid];
            for(size_t i = 0; i < lag_set.size(); i++) {
                size_t lag_i = lag_set[i];
                y[i] = lagged_inner_product(univate_series, start, end, static_cast<size_t>(0), lag_i);
                for(size_t j = i; j < lag_set.size(); j++) {
                    size_t lag_j = lag_set[j];
                    double Hij = lagged_inner_product(univate_series, start, end, lag_i, lag_j);
                    Hessian.at(i, j) = Hij;
                }
            }
            for(size_t i = 0; i < lag_set.size(); i++) {
                for(size_t j = 0; j < i; j++) {
                    Hessian.at(i, j) = Hessian.at(j, i);
                }
                Hessian.at(i, i) += lambda;
            }
            ls_solve_chol(Hessian.data(), y.data(), lag_set.size());
        }
    }

    double fun(dmat_t& lag_val) {
        size_t start = lag_set.back();
        size_t end = m;
        double loss = 0.0;
        for(size_t t = 0; t < k; t++) {
            double local_loss = 0.0;
            for(size_t i = start; i < end; i++) {
                double err =  -T.at(i, t);
                for(size_t l = 0; l < lag_set.size(); l++) {
                    err += lag_val.at(l, t) * T.at(i - lag_set[l], t);
                }
                local_loss += err * err;
            }
            loss += local_loss;
        }
        loss *= 0.5;
        double reg = 0.5 * do_dot_product(lag_val, lag_val);
        return reg * lambda + loss;
    }
};

template<typename val_type>
arr_solver<val_type>::arr_solver(arr_prob_t *prob, arr_param_t *param):
    prob(prob), param(param), fun_obj(NULL), tron_obj(NULL), solver_obj(NULL), done_init(false) { // {{{
    if(prob->lag_set != NULL) {
        switch(param->solver_type) {
            case ARR_LS_FULL:
                fun_obj = new arr_ls_fY_IX<val_type>(prob, param);
                break;
            case ARR_LS_MISSING:
                fun_obj = new arr_ls_pY_IX<val_type>(prob, param);
                break;
            default:
                fprintf(stderr, "Solver not supported\n");
                break;
        }
        fflush(stdout);
        int max_cg_iter = param->max_cg_iter;
        if(max_cg_iter >= fun_obj->get_nr_variable()) {
            max_cg_iter = fun_obj->get_nr_variable();
        }
        bool pure_cg = true; // as we have only quadratic problems here
        tron_obj = new TRON<val_type>(fun_obj, param->eps, param->eps_cg, param->max_tron_iter, max_cg_iter, pure_cg);
        tron_obj->set_print_string(get_print_fun(param));
    } else {
        // fall back to the simple matrix factorization
        switch(param->solver_type) {
            case ARR_LS_FULL:
                solver_obj = new l2r_ls_fY_IX_chol<val_type>(*(prob->Y), *(prob->H), param->lambdaI);
                break;
            case ARR_LS_MISSING:
                solver_obj = new l2r_ls_pY_IX_chol<val_type>(*(prob->Y), *(prob->H), param->lambdaI);
                break;
            default:
                fprintf(stderr, "Solver not supported\n");
                break;
        }
    }
} //}}}


void trmf_initialization(const trmf_prob_t& prob, const trmf_param_t& param, dmat_t& W, dmat_t& H, dmat_t& lag_val) {
    size_t m = prob.m;
    size_t n = prob.n;
    size_t k = prob.k;
    rng_t rng;
    W = dmat_t::rand(rng, m, k, 0, 1, ROWMAJOR);
    H = dmat_t::rand(rng, n, k, 0, 1, ROWMAJOR);
    lag_val = dmat_t::randn(rng, prob.lag_set->size(), k, 0, 1, COLMAJOR);

    W.grow_body();
    H.grow_body();
    lag_val.grow_body();
}

bool check_dimension(const trmf_prob_t& prob, const trmf_param_t& param, const dmat_t& W, const dmat_t& H, const dmat_t& lag_val) {
    bool pass = true;
    if(prob.m != W.rows) {
        fprintf(stderr, "[ERR MSG]: Y.rows (%ld) != W.rows (%ld)\n", prob.m, W.rows);
        pass = false;
    }
    if(prob.n != H.rows) {
        fprintf(stderr, "[ERR MSG]: Y.cols (%ld) != H.rows (%ld)\n", prob.n, H.rows);
        pass = false;
    }
    if(W.cols != H.cols) {
        fprintf(stderr, "[ERR MSG]: W.cols (%ld) != H.cols (%ld)\n", W.cols, H.cols);
        pass = false;
    }
    if(prob.lag_set->size() != lag_val.rows) {
        fprintf(stderr, "[ERR MSG]: lag_set.size(%ld) != lag_val.rows(%ld)\n", prob.lag_set->size(), lag_val.rows);
        pass = false;
    }
    if(W.cols != lag_val.cols) {
        fprintf(stderr, "[ERR MSG]: W.cols(%ld) != lag_val.cols(%ld)\n", W.cols, lag_val.cols);
        pass = false;
    }
    if(!W.is_rowmajor()) {
        fprintf(stderr, "[ERR MSG]: W should be rowmajored\n");
        pass = false;
    }
    if(!H.is_rowmajor()) {
        fprintf(stderr, "[ERR MSG]: H should be rowmajored\n");
        pass = false;
    }
    if(!lag_val.is_colmajor()) {
        fprintf(stderr, "[ERR MSG]: lag_val should be colmajored\n");
        pass = false;
    }
    return pass;
}

// W and H must be ROWMAJOR. and lag_val must be COLMAJOR
void trmf_train(trmf_prob_t& prob, trmf_param_t& param, dmat_t& W, dmat_t& H, dmat_t& lag_val) {
    ivec_t& lag_set = *(prob.lag_set);
    gmat_t& Y = *(prob.Y);
    gmat_t& Yt = *(prob.Yt); // transpose of Y
    if(param.max_tron_iter > 1) {
        param.max_cg_iter *= param.max_tron_iter;
        param.max_tron_iter = 1;
    }
    if(param.verbose > 0) {
        fprintf(stdout, ">> param.solver_type %d\n", param.solver_type);
        fprintf(stdout, ">> param.max_iter %d\n", param.max_iter);
        fprintf(stdout, ">> param.lambdaI %g\n", param.lambdaI);
        fprintf(stdout, ">> param.lambdaAR %g\n", param.lambdaAR);
        fprintf(stdout, ">> param.lambdaLag %g\n", param.lambdaLag);
        fprintf(stdout, ">> param.period_W %d\n", param.period_W);
        fprintf(stdout, ">> param.period_H %d\n", param.period_H);
        fprintf(stdout, ">> param.period_Lag %d\n", param.period_Lag);
        fprintf(stdout, ">> param.threads %d\n", param.threads);
        fprintf(stdout, ">> param.verbose %d\n", param.verbose);
        fprintf(stdout, ">> param.eps %g\n", param.eps);
        fprintf(stdout, ">> param.eps_cg %g\n", param.eps_cg);
        fprintf(stdout, ">> param.max_tron_iter %d\n", param.max_tron_iter);
        fprintf(stdout, ">> param.max_cg_iter %d\n", param.max_cg_iter);
        fprintf(stdout, ">> prob.lag_size %ld:  ", lag_set.size());

        for(size_t idx = 0; idx < lag_set.size(); idx++) {
            fprintf(stdout, " %d", lag_set[idx]);
        }
        fprintf(stdout, "\n");
        fflush(stdout);
    }

    // check dimension
    if(!check_dimension(prob, param, W, H, lag_val)) {
        return;
    }

    omp_set_num_threads(param.threads);

    arr_prob_t subprob_W(&Y, &H, &lag_set, &lag_val);
    arr_prob_t subprob_H(&Yt, &W, NULL, NULL);
    arr_solver<ValueType> W_solver(&subprob_W, &param);
    arr_solver<ValueType> H_solver(&subprob_H, &param);
    l2r_autoregressive_solver<ValueType> LV_solver(W, lag_set, param.lambdaLag);


	walltime_clock_t timer;
    double Wtime=0, Htime=0, LVtime=0;
    for(int iter = 1; iter <= param.max_iter; iter++) {

        if(param.verbose > 0) {
            //fprintf(stdout, "TRMF-iter %d\n", iter);
            fflush(stdout);
        }

        if((iter % param.period_H) == 0)
        {
            timer.tic();
            H_solver.init_prob();
            H_solver.solve(H.data());
            Htime += timer.toc();
            if(param.verbose) {
                fprintf(stderr, ">> iter %d F %g\n", iter, do_dot_product(H, H));
            }
        }

        if((iter % param.period_W) == 0)
        {
            timer.tic();
            W_solver.init_prob();
            W_solver.solve(W.data());
            Wtime += timer.toc();
            if(param.verbose) {
                fprintf(stderr, ">> iter %d X %g\n", iter, do_dot_product(W, W));
            }
        }


        if((iter % param.period_Lag) == 0)
        {
            if(param.verbose) {
                fprintf(stderr, ">> iter %d LV(%ld %ld) %g\n", iter, lag_val.rows, lag_val.cols, do_dot_product(lag_val, lag_val));
            }
            timer.tic();
            LV_solver.init_prob();
            LV_solver.solve(lag_val);
            LVtime += timer.toc();
            if(param.verbose) {
                fprintf(stderr, ">> iter %d LV %g\n", iter, do_dot_product(lag_val, lag_val));
            }
        }
        if(param.verbose > 0) {
            //fprintf(stdout, " Wtime %g Htime %g LVtime %g\n", Wtime, Htime, LVtime);
        }
    }
}

void c_trmf_train(const PyMatrix *pyY, uint32_t *py_lag_set, uint32_t py_lag_size,
        PyMatrix *pyW, PyMatrix *pyH, PyMatrix *pylag_val, int warm_start,
        double lambdaI, double lambdaAR, double lambdaLag,
        int32_t max_iter, int32_t period_W, int32_t period_H, int32_t period_Lag,
        int32_t threads, int32_t missing, int32_t verbose) {
    gmat_wrapper_t Y(pyY), W(pyW), H(pyH), lag_val(pylag_val);
    ivec_t lag_set(static_cast<size_t>(py_lag_size), py_lag_set);
    gmat_wrapper_t Yt = Y.transpose();
    size_t k = W.get_gmat().cols;

    trmf_prob_t prob(&Y.get_gmat(), &Yt.get_gmat(), &lag_set, k);
    trmf_param_t param;
    param.lambdaI = lambdaI;
    param.lambdaAR = lambdaAR;
    param.lambdaLag = lambdaLag;
    param.max_iter = max_iter;
    param.period_W = period_W;
    param.period_H = period_H;
    param.period_Lag = period_Lag;
    param.threads = threads;
    param.verbose = verbose;
    param.solver_type = (missing != 0) ? ARR_LS_MISSING: ARR_LS_FULL;

    if(!warm_start) {
        trmf_initialization(prob, param, W.get_dense(), H.get_dense(), lag_val.get_dense());
    }
    trmf_train(prob, param, W.get_dense(), H.get_dense(), lag_val.get_dense());
    //printf("|w|=%g |h|=%g \n", do_dot_product(W.get_dense(), W.get_dense()), do_dot_product(H.get_dense(), H.get_dense()));
    return ;
}




