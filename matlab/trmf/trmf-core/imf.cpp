
#include "imf.h"
#include "smat.h"
#include "dmat.h"
#include "dbilinear.h"

static double norm(double *W, size_t size) {
	double ret = 0;
	for(size_t i = 0; i < size; i++)
		ret += W[i]*W[i];
	return sqrt(ret);
}

void arr_mf_train(arr_mf_prob_t *prob, arr_mf_param_t *param, double *W, double *H, smat_t *testY, double *rmse) { // {{{
	//size_t m = prob->m, n = prob->n;
	size_t k = param->k;

	omp_set_num_threads(param->threads);

	smat_t &Y = *(prob->Y);
	smat_t Yt = Y.transpose();

	arr_prob_t subprob_w(&Yt, H, k, 0, NULL, NULL);
	arr_prob_t subprob_h(&Y, W, k, prob->lag_size, prob->lag_idx, prob->lag_val);
	arr_solver W_solver(&subprob_w, param);
	arr_solver H_solver(&subprob_h, param);


	if(param->verbose != 0) {
		//printf("|W0| (%ld %ld)= %.6f\n", k, m, norm(W,m*k));
		//printf("|H0| (%ld %ld)= %.6f\n", k, n, norm(H,n*k));
	}

	double Wtime=0, Htime=0, start_time=0;
	for(int iter = 1; iter <= param->maxiter; iter++) {
		start_time = omp_get_wtime();
		//printf("F %g", do_dot_product(W,W,k*prob->m));
		W_solver.init_prob();
		W_solver.solve(W);
		//printf("-> %g\n", do_dot_product(W,W,k*prob->m));
		Wtime += omp_get_wtime()-start_time;

		start_time = omp_get_wtime();
		//printf("X %g", do_dot_product(H,H,k*prob->n));
		H_solver.init_prob();
		H_solver.solve(H);
		//printf("X-> %g", do_dot_product(H,H,k*prob->n));
		Htime += omp_get_wtime()-start_time;
		if(param->verbose != 0) {
			//printf("IMF-iter %d W %.5g H %.5g walltime %.5g", iter, Wtime, Htime, Wtime+Htime);
			fflush(stdout);
			/*
			double reg_w = 0.5*param->lambdaI*do_dot_product(W, W, k*prob->m);
			W_solver.init_prob();
			double loss = W_solver.fun(W) - reg_w;
			H_solver.init_prob();
			double reg_h = H_solver.fun(H) - loss;
			double obj = loss + reg_w + reg_h;
			printf(" loss %g reg_w %g reg_h %g obj %g", loss, reg_w, reg_h, obj);
			if(testY) {
				double tmp_rmse = cal_rmse(*testY,W,H,k);
				if(rmse!=NULL) *rmse = tmp_rmse;
				printf(" rmse %lf", tmp_rmse);
			}
			*/
			//puts("");
			fflush(stdout);
		}
	}
}// }}}
