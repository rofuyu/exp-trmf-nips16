#ifndef RF_TRON_H
#define RF_TRON_H


#include <stdarg.h>
#include <stddef.h>
#include "rf_matrix.h" // to include BLAS/LAPACK header

enum {GD_LS=0, TRON_TR=1, TRON_LS=2};

static void default_print(const char *buf);

class function { // {{{
    public:
        virtual double fun(void *w) = 0 ;
        virtual void grad(void *w, void *g) = 0 ;
        virtual void Hv(void *s, void *Hs) = 0 ;
        virtual double line_search(void *s, void *w, void *g,
                double init_step_size, double *fnew, bool do_update=true) { return 0; }
        virtual bool line_search_supported() { return false; }

        virtual int get_nr_variable(void) = 0 ;
        virtual ~function(void){}
        virtual void init(){}
}; // }}}

template<typename val_type>
class TRON { // {{{
    public:
        TRON(const function *fun_obj, double eps = 0.1, double eps_cg = 0.1, size_t max_iter = 100, size_t max_cg_iter = 20, bool pure_cg = false);
        ~TRON();

        void tron(val_type *w, bool set_w_to_zero = true, int solver_descend_type = TRON_TR);
        void tron_trustregion(val_type *w, bool set_w_to_zero = true);
        void tron_linesearch(val_type *w, bool set_w_to_zero = true);
        void gd_linesearch(val_type *w, bool set_w_to_zero = true);
        void set_print_string(void (*i_print) (const char *buf));
        void set_eps(val_type eps, val_type eps_cg = 0.1) {this->eps = eps; this->eps_cg = eps_cg;}

    private:
        int trcg(double delta, val_type *g, val_type *s, val_type *r, double *cg_rnorm);
        double norm_inf(size_t n, val_type *x);

        double eps;
        double eps_cg;
        size_t max_iter;
        size_t max_cg_iter;
        bool pure_cg;
        function *fun_obj;
        void info(const char *fmt,...);
        void (*tron_print_string)(const char *buf);
        // local variables for tron
        val_type *s, *r, *w_new, *g;
        // local variables for trcg
        val_type *d, *Hd;

}; // }}}


// ------------- Implementation ------------------------

static void default_print(const char *buf) { // {{{
    fputs(buf,stdout);
    fflush(stdout);
} // }}}

template<typename val_type>
void TRON<val_type>::info(const char *fmt,...) { // {{{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf,fmt,ap);
    va_end(ap);
    (*tron_print_string)(buf);
} // }}}

template<typename val_type>
TRON<val_type>::TRON(const function *fun_obj, double eps, double eps_cg, size_t max_iter, size_t max_cg_iter, bool pure_cg) { // {{{
    this->fun_obj=const_cast<function *>(fun_obj);
    this->eps=eps;
    this->eps_cg=eps_cg;
    this->max_iter=max_iter;
    this->max_cg_iter = max_cg_iter;
    this->pure_cg = pure_cg;
    tron_print_string = default_print;
    ptrdiff_t n = this->fun_obj->get_nr_variable();
    s = CALLOC(val_type, n);
    r = CALLOC(val_type, n);
    w_new = CALLOC(val_type, n);
    g = CALLOC(val_type, n);
    d = CALLOC(val_type, n);
    Hd = CALLOC(val_type, n);
    /*
    s = new val_type[n];
    r = new val_type[n];
    w_new = new val_type[n];
    g = new val_type[n];
    d = new val_type[n];
    Hd = new val_type[n];
    */
} // }}}

template<typename val_type>
TRON<val_type>::~TRON() { // {{{
    free(s);
    free(r);
    free(w_new);
    free(g);
    free(d);
    free(Hd);
    /*
    delete[] g;
    delete[] r;
    delete[] w_new;
    delete[] s;
    delete[] d;
    delete[] Hd;
    */
} // }}}

template<typename val_type>
void TRON<val_type>::tron(val_type *w, bool set_w_to_zero, int solver_descend_type) { // {{{
    // tron_obj->tron(w, true);// zero-initization for w
    if(solver_descend_type == TRON_TR || fun_obj->line_search_supported() == false)
        TRON<val_type>::tron_trustregion(w, set_w_to_zero);
    else {
        if(solver_descend_type == TRON_LS)
            TRON<val_type>::tron_linesearch(w, set_w_to_zero);
        else if(solver_descend_type == GD_LS)
            TRON<val_type>::gd_linesearch(w, set_w_to_zero);
    }
} // }}}

template<typename val_type>
void TRON<val_type>::tron_trustregion(val_type *w, bool set_w_to_zero) { // {{{
    // Parameters for updating the iterates.
    double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

    // Parameters for updating the trust region size delta.
    double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

    ptrdiff_t n = fun_obj->get_nr_variable();
    int i, cg_iter;
    double delta, snorm;
    val_type one=1.0;
    double alpha, f, fnew, prered, actred, gs;
    size_t search = 1, iter = 1;
    ptrdiff_t inc = 1;

    if (set_w_to_zero)
        for (i=0; i<n; i++)
            w[i] = 0;

    f = fun_obj->fun(w);

    //fprintf(stderr, "fun is done\n");

    fun_obj->grad(w, g);
    //fprintf(stderr, "grad is done\n");
    //fprintf(stderr, "TRON23: max_iter %ld cg_iter %ld, eps %g, eps_cg %g\n", this->max_iter, this->max_cg_iter, this->eps, this->eps_cg);
    //fprintf(stderr, "n %ld max_iter %ld max_cg_iter %ld\n", n, max_iter, max_cg_iter);
    //delta = dnrm2_(&n, g, &inc);
    //delta = sqrt(ddot_(&n, g, &inc, g, &inc));
    delta = sqrt(dot(&n, g, &inc, g, &inc));
    double gnorm1 = delta;
    double gnorm = gnorm1;

    //if (gnorm <= eps*gnorm1 || gnorm1 < eps) {
    if (gnorm <= eps*gnorm1) {
        search = 0;
    }

    iter = 1;


    while (iter <= max_iter && search)
    {
        double cg_rnorm=0;
        cg_iter = trcg(delta, g, s, r, &cg_rnorm);

        //memcpy(w_new, w, sizeof(double)*n);
        //dcopy_(&n, w, &inc, w_new, &inc);
        copy(&n, w, &inc, w_new, &inc);
        //daxpy_(&n, &one, s, &inc, w_new, &inc);
        axpy(&n, &one, s, &inc, w_new, &inc);

        //gs = ddot_(&n, g, &inc, s, &inc);
        gs = dot(&n, g, &inc, s, &inc);
        //prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
        prered = -0.5*(gs-dot(&n, s, &inc, r, &inc));
        fnew = fun_obj->fun(w_new);

        // Compute the actual reduction.
        actred = f - fnew;

        // On the first iteration, adjust the initial step bound.
        //snorm = dnrm2_(&n, s, &inc);
        //snorm = sqrt(ddot_(&n, s, &inc, s, &inc));
        snorm = sqrt(dot(&n, s, &inc, s, &inc));
        if (iter == 1)
            delta = std::min(delta, snorm);

        // Compute prediction alpha*snorm of the step.
        if (fnew - f - gs <= 0)
            alpha = sigma3;
        else
            alpha = std::max(sigma1, -0.5*(gs/(fnew - f - gs)));

        // Update the trust region bound according to the ratio of actual to predicted reduction.
        if (actred < eta0*prered)
            delta = std::min(std::max(alpha, sigma1)*snorm, sigma2*delta);
        else if (actred < eta1*prered)
            delta = std::max(sigma1*delta, std::min(alpha*snorm, sigma2*delta));
        else if (actred < eta2*prered)
            delta = std::max(sigma1*delta, std::min(alpha*snorm, sigma3*delta));
        else
            delta = std::max(delta, std::min(alpha*snorm, sigma3*delta));

        info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d |g| %5.3e\n", iter, actred, prered, delta, f, gnorm, cg_iter, cg_rnorm);
        //info("iter %2d act %5.3e pre %5.3e delta %5.3e f %.17g |g| %.17g CG %3d |g| %5.3e\n", iter, actred, prered, delta, f, gnorm, cg_iter, cg_rnorm);

        if (actred > eta0*prered)
        {
            iter++;
            //memcpy(w, w_new, sizeof(double)*n);
            //dcopy_(&n, w_new, &inc, w, &inc);
            copy(&n, w_new, &inc, w, &inc);
            f = fnew;
            fun_obj->grad(w, g);

            //gnorm = dnrm2_(&n, g, &inc);
            //gnorm = sqrt(ddot_(&n, g, &inc, g, &inc));
            gnorm = sqrt(dot(&n, g, &inc, g, &inc));
            if (gnorm <= eps*gnorm1)
                break;
        }
        if (f < -1.0e+32)
        {
            info("WARNING: f < -1.0e+32\n");
            break;
        }
        if (fabs(actred) <= 0 && prered <= 0)
        {
            info("WARNING: actred and prered <= 0\n");
            break;
        }
        if (fabs(actred) <= 1.0e-12*fabs(f) &&
            fabs(prered) <= 1.0e-12*fabs(f))
        {
            info("WARNING: actred and prered too small\n");
            break;
        }
    }
} // }}}

template<typename val_type>
void TRON<val_type>::tron_linesearch(val_type *w, bool set_w_to_zero) { // {{{
    ptrdiff_t n = fun_obj->get_nr_variable();
    int i;
    val_type step_size=1.0;
    double f, fnew, actred;
    double init_step_size = 1;
    const double delta=0; // delta = 0 => trcg reduces to standard CG
    //val_type one=1.0;
    size_t search = 1, iter = 1, cg_iter = 0;
    ptrdiff_t inc = 1;

    if (set_w_to_zero)
        for (i=0; i<n; i++)
            w[i] = 0;
    // calculate gradient norm at w=0 for stopping condition
#pragma omp parallel for schedule(static)
    for(i=0;i<n;i++)
        w_new[i] = 0;
    f = fun_obj->fun(w_new);
    fun_obj->grad(w_new, g);
    double gnorm0 = sqrt(dot(&n, g, &inc, g, &inc));

    if (!set_w_to_zero) {
        f = fun_obj->fun(w);
        fun_obj->grad(w, g);
    }
    double gnorm = sqrt(dot(&n, g, &inc, g, &inc));

    //if (gnorm <= eps*gnorm1 || gnorm1 < eps)
    if (gnorm <= eps*gnorm0)
        search = 0;

    iter = 1;

    bool do_update = true; // perform w/grad/Hv updates inside line_search
    while (iter <= max_iter && search)
    {
        double cg_rnorm=0;
        cg_iter = trcg(delta, g, s, r, &cg_rnorm);

        step_size = fun_obj->line_search(s, w, g, init_step_size, &fnew, do_update);

        actred = f - fnew;
        if(step_size == 0) {
            info("WARNING: line search fails\n");
            break;
        }

        if(!do_update) {
            //daxpy_(&n, &step_size, s, &inc, w, &inc);
            axpy(&n, &step_size, s, &inc, w, &inc);
        }

        info("iter %2d f %5.3e |g| %5.3e CG %3d step_size %5.3e |g| %5.3e\n", iter, f, gnorm, cg_iter, step_size, cg_rnorm);

        f = fnew;
        iter++;

        if(!do_update) fun_obj->fun(w);
        fun_obj->grad(w, g);

        gnorm = sqrt(dot(&n, g, &inc, g, &inc));

        if (gnorm <= eps*gnorm0)
            break;
        if (f < -1.0e+32) {
            info("WARNING: f < -1.0e+32\n");
            break;
        }
        if (fabs(actred) <= 1.0e-12*fabs(f)) {
            info("WARNING: actred too small\n");
            break;
        }
    }
} // }}}

template<typename val_type>
void TRON<val_type>::gd_linesearch(val_type *w, bool set_w_to_zero) { // {{{
    ptrdiff_t n = fun_obj->get_nr_variable();
    int i;
    val_type step_size=1.0;
    double f, fnew, actred;
    double init_step_size = 1;
    //const double delta=0; // delta = 0 => trcg reduces to standard CG
    //val_type one=1.0;
    size_t search = 1, iter = 1;
    ptrdiff_t inc = 1;

    if (set_w_to_zero)
        for (i=0; i<n; i++)
            w[i] = 0;
    // calculate gradient norm at w=0 for stopping condition
#pragma omp parallel for schedule(static)
    for(i=0;i<n;i++)
        w_new[i] = 0;
    f = fun_obj->fun(w_new);
    fun_obj->grad(w_new, g);
    double gnorm0 = sqrt(dot(&n, g, &inc, g, &inc));

    if (!set_w_to_zero) {
        f = fun_obj->fun(w);
        fun_obj->grad(w, g);
    }
    double gnorm = sqrt(dot(&n, g, &inc, g, &inc));

    //if (gnorm <= eps*gnorm1 || gnorm1 < eps)
    if (gnorm <= eps*gnorm0)
        search = 0;

    iter = 1;

    bool do_update = true; // perform w/grad/Hv updates inside line_search
    while (iter <= (max_iter*max_cg_iter) && search)
    {
        //double cg_rnorm=0;
        //cg_iter = trcg(delta, g, s, r, &cg_rnorm);

#pragma omp parallel for
        for (i=0; i<n; i++)
            s[i] = -g[i];

        step_size = fun_obj->line_search(s, w, g, init_step_size, &fnew, do_update);

        actred = f - fnew;
        if(step_size == 0) {
            info("WARNING: line search fails\n");
            break;
        }

        if(!do_update)
            axpy(&n, &step_size, s, &inc, w, &inc);

        info("iter %2d f %5.3e |g| %5.3e step_size %5.3e\n", iter, f, gnorm, step_size);

        f = fnew;
        iter++;

        if(!do_update) fun_obj->fun(w);
        fun_obj->grad(w, g);

        gnorm = sqrt(dot(&n, g, &inc, g, &inc));

        if (gnorm <= eps*gnorm0)
            break;
        if (f < -1.0e+32) {
            info("WARNING: f < -1.0e+32\n");
            break;
        }
        if (fabs(actred) <= 1.0e-12*fabs(f)) {
            info("WARNING: actred too small\n");
            break;
        }
    }
} // }}}

template<typename val_type>
int TRON<val_type>::trcg(double delta, val_type *g, val_type *s, val_type *r, double *cg_rnorm) { // {{{
    int i;
    ptrdiff_t n = fun_obj->get_nr_variable();
    ptrdiff_t inc = 1;
    val_type one = 1;
    /*
    double *d = new double[n];
    double *Hd = new double[n];
    */
    val_type rTr, rnewTrnew, alpha, beta, cgtol;

#pragma omp parallel for
    for (i=0; i<n; i++)
    {
        s[i] = 0;
        r[i] = -g[i];
        d[i] = r[i];
    }
    //cgtol = 0.1*dnrm2_(&n, g, &inc);
    //cgtol = 0.1*sqrt(ddot_(&n, g, &inc, g, &inc));
    //cgtol = eps*sqrt(ddot_(&n, g, &inc, g, &inc));
    cgtol = eps_cg*sqrt(dot(&n, g, &inc, g, &inc));
    //cgtol = eps*sqrt(dot(&n, g, &inc, g, &inc));

    size_t cg_iter = 0;
    //rTr = ddot_(&n, r, &inc, r, &inc);
    rTr = dot(&n, r, &inc, r, &inc);
    //double rTr_init = rTr;
    while (1)
    {
        //*cg_rnorm = sqrt(ddot_(&n, r, &inc, r, &inc));
        *cg_rnorm = sqrt(dot(&n, r, &inc, r, &inc));
        if (*cg_rnorm <= cgtol)
            break;

        /*
		*cg_rnorm = sqrt(rTr);
		if((rTr < eps_cg * rTr_init) && (rTr < eps_cg))
			break;
        */

        if (max_cg_iter > 0 && cg_iter >= max_cg_iter)
            break;
        cg_iter++;
        fun_obj->Hv(d, Hd);

        //alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
        alpha = rTr/dot(&n, d, &inc, Hd, &inc);
        //daxpy_(&n, &alpha, d, &inc, s, &inc);
        axpy(&n, &alpha, d, &inc, s, &inc);
        //if (sqrt(ddot_(&n, s, &inc, s, &inc)) > delta)
        if (!pure_cg && delta > 0 && sqrt(dot(&n, s, &inc, s, &inc)) > delta)
        {
            info("cg reaches trust region boundary\n");
            alpha = -alpha;
            //daxpy_(&n, &alpha, d, &inc, s, &inc);
            axpy(&n, &alpha, d, &inc, s, &inc);

            //double std = ddot_(&n, s, &inc, d, &inc);
            double std = dot(&n, s, &inc, d, &inc);
            //double sts = ddot_(&n, s, &inc, s, &inc);
            double sts = dot(&n, s, &inc, s, &inc);
            //double dtd = ddot_(&n, d, &inc, d, &inc);
            double dtd = dot(&n, d, &inc, d, &inc);
            double dsq = delta*delta;
            double rad = sqrt(std*std + dtd*(dsq-sts));
            if (std >= 0)
                alpha = (dsq - sts)/(std + rad);
            else
                alpha = (rad - std)/dtd;
            //daxpy_(&n, &alpha, d, &inc, s, &inc);
            axpy(&n, &alpha, d, &inc, s, &inc);
            alpha = -alpha;
            //daxpy_(&n, &alpha, Hd, &inc, r, &inc);
            axpy(&n, &alpha, Hd, &inc, r, &inc);
            break;
        }
        alpha = -alpha;
        //daxpy_(&n, &alpha, Hd, &inc, r, &inc);
        axpy(&n, &alpha, Hd, &inc, r, &inc);
        //rnewTrnew = ddot_(&n, r, &inc, r, &inc);
        rnewTrnew = dot(&n, r, &inc, r, &inc);
        beta = rnewTrnew/rTr;
        //dscal_(&n, &beta, d, &inc);
        val_type tmp = beta - (val_type)1.0;
        //daxpy_(&n, &tmp, d, &inc, d, &inc);
        axpy(&n, &tmp, d, &inc, d, &inc);
        //daxpy_(&n, &one, r, &inc, d, &inc);
        axpy(&n, &one, r, &inc, d, &inc);
        rTr = rnewTrnew;
    }
    return(cg_iter);
} // }}}

template<typename val_type>
double TRON<val_type>::norm_inf(size_t n, val_type *x) { // {{{
    double dmax = fabs(x[0]);
    for (int i=1; i<n; i++)
        if (fabs(x[i]) >= dmax)
            dmax = fabs(x[i]);
    return(dmax);
} // }}}

template<typename val_type>
void TRON<val_type>::set_print_string(void (*print_string) (const char *buf)) { // {{{
    tron_print_string = print_string;
} // }}}

#endif // end of RF_TRON_H


