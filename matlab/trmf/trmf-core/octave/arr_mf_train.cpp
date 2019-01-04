

#include "mex.h"
#include "../dbilinear.h"
#include "../imf.h"
#include <omp.h>
#include <cstring>

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL


void exit_with_help()
{
	mexPrintf(
	"Usage: [W' H' rmse walltime] = arr_mf_train(Y, testY, lag_idx, lag_val [, 'options'])\n"
	"Usage: [W' H' rmse walltime] = arr_mf_train(Y, testY, lag_idx, lag_val, W', H' [, 'options'])\n"
	" lag_idx : a length lag_size array of lag index\n"
	" lag_val : a k by lag_size array of lag value\n"
	"options:\n"
	"    -s type : set type of solver (default 20)\n"
	"       30: ARR_LS_FULL    (0.5*lambdaI*{|W|_F^2+|H|+F^2} + 0.5*lambdaAR*{AR(lag_idx, lag_val, H)} + 0.5*squared-L2-loss)\n"
	"       31: ARR_LS_MISSING (0.5*lambdaI*{|W|_F^2+|H|+F^2} + 0.5*lambdaAR*{AR(lag_idx, lag_val, H)} + 0.5*squared-L2-loss)\n"
	"    -n threads : set the number of threads (default 4)\n"
	"    -k rank : set the rank (default 10)\n"
	"    -li lambdaI : set the lambdaI\n"
	"    -la lambdaAR : set the lambdaAR\n"
	"    -e epsilon : set stopping criterion epsilon of tron (default 0.1)\n"
	"    -t max_iter: set the number of iterations (default 10)\n"
	"    -T max_tron_iter: set the number of iterations used in TRON (default 3)\n"
	"    -g max_cg_iter: set the number of iterations used in CG (default 20)\n"
	"    -q verbose: show information or not (default 1)\n"
	);
}

arr_mf_param_t parse_command_line(int nrhs, const mxArray *prhs[])
{
	arr_mf_param_t param;   // default values have been set by the constructor
	param.verbose = 1;
	int i, argc = 1;
	int option_pos = -1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];

	if(nrhs == 4 or nrhs == 6)
		return param;
	if(nrhs == 5 or nrhs == 7)
		option_pos = nrhs-1;

	// put options in argv[]
	if(option_pos>0)
	{
		mxGetString(prhs[option_pos], cmd,  mxGetN(prhs[option_pos]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.max_tron_iter = atoi(argv[i]);
				break;

			case 'g':
				param.max_cg_iter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'l':
				switch (argv[i-1][2]) {
					case 'a':
						param.lambdaAR = atof(argv[i]);
						break;
					case 'i':
					case '\0':
					default:
						param.lambda = param.lambdaI = atof(argv[i]);
						break;
				}
				break;
			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			default:
				mexPrintf("unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	// Squared-L2 loss, use CG is enough
	if(param.solver_type % 10 == 0 or param.solver_type % 10 == 1) {
		param.max_cg_iter = param.max_tron_iter*param.max_cg_iter;
		param.max_tron_iter = 1;
	}
	omp_set_num_threads(param.threads);

	return param;
}

bool isDoubleSparse(const mxArray *mxM) {
		if(!mxIsDouble(mxM)) {
			mexPrintf("Error: matrix must be double\n");
			return false;
		}

		if(!mxIsSparse(mxM)) {
			mexPrintf("matrix must be sparse; "
					"use sparse(matrix) first\n");
			return false;
		}
		return true;
}
bool isDoubleDense(const mxArray *mxM) {
		if(!mxIsDouble(mxM)) {
			mexPrintf("Error: matrix must be double\n");
			return false;
		}

		if(mxIsSparse(mxM)) {
			mexPrintf("matrix must be dense; "
					"use full(matrix) first\n");
			return false;
		}
		return true;
}
static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

int transpose(const mxArray *M, mxArray **Mt) {
	mxArray *prhs[1] = {const_cast<mxArray *>(M)}, *plhs[1];
	if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
	{
		mexPrintf("Error: cannot transpose training instance matrix\n");
		return -1;
	}
	*Mt = plhs[0];
	return 0;
}

// convert matlab sparse matrix to C smat fmt

class mxSparse_iterator_t: public entry_iterator_t {
	private:
		mxArray *Mt;
		mwIndex *ir_t, *jc_t;
		double *v_t;
		size_t	rows, cols, cur_idx, cur_row;
	public:
		mxSparse_iterator_t(const mxArray *M){
			rows = mxGetM(M); cols = mxGetN(M);
			nnz = *(mxGetJc(M) + cols);
			transpose(M, &Mt);
			ir_t = mxGetIr(Mt); jc_t = mxGetJc(Mt); v_t = mxGetPr(Mt);
			cur_idx = cur_row = 0;
		}
		rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			while (cur_idx >= jc_t[cur_row+1]) ++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			rate_t ret(cur_row, ir_t[cur_idx], v_t[cur_idx]);
			cur_idx++;
			return ret;
		}
		~mxSparse_iterator_t(){
			mxDestroyArray(Mt);
		}

};

class mxCoo_iterator_t: public entry_iterator_t {
	private:
		double *row_idx, *col_idx, *val;
		size_t cur_idx;
	public:
		size_t rows, cols;
		mxCoo_iterator_t(const mxArray *M, size_t rows=0, size_t cols=0): rows(rows), cols(cols){
			double *data = mxGetPr(M);
			nnz = mxGetM(M);
			row_idx = &data[0];
			col_idx = &data[nnz];
			val = &data[2*nnz];
			cur_idx = 0;
			for(size_t idx = 0; idx < nnz; idx++) {
				if((size_t)row_idx[idx] > rows) rows = (size_t) row_idx[idx];
				if((size_t)col_idx[idx] > cols) cols = (size_t) col_idx[idx];
			}
		}
		rate_t next() {
			size_t row = (size_t) row_idx[cur_idx], col = (size_t) col_idx[cur_idx];
			rate_t ret(row-1, col-1, val[cur_idx]);
			cur_idx++;
			return ret;
		}
};

class mxDense_iterator_t: public entry_iterator_t {
	private:
		size_t cur_idx;
		double *val;
	public:
		size_t rows, cols;
		mxDense_iterator_t(const mxArray *mxM): rows(mxGetM(mxM)), cols(mxGetN(mxM)), val(mxGetPr(mxM)){
			cur_idx = 0; nnz = rows*cols;
		}
		rate_t next() {
			rate_t ret(cur_idx%cols, cur_idx/cols, val[cur_idx]);
			cur_idx++;
			return ret;
		}
};

smat_t mxDense_to_smat(const mxArray *mxM, smat_t &R) {
	mxDense_iterator_t entry_it(mxM);
	R.load_from_iterator(entry_it.rows, entry_it.cols, entry_it.nnz, &entry_it);
	return R;
}

smat_t mxCoo_to_smat(const mxArray *mxM, smat_t &R, size_t rows, size_t cols) {
	mxCoo_iterator_t entry_it(mxM, rows, cols);
	R.load_from_iterator(entry_it.rows, entry_it.cols, entry_it.nnz, &entry_it);
	return R;
}

smat_t mxSparse_to_smat(const mxArray *mxM, smat_t &R) {
	long rows = mxGetM(mxM), cols = mxGetN(mxM), nnz = *(mxGetJc(mxM) + cols);
	mxSparse_iterator_t entry_it(mxM);
	R.load_from_iterator(rows, cols, nnz, &entry_it);
	return R;
}

smat_t mxArray_to_smat(const mxArray *mxM, smat_t &R, size_t rows=0, size_t cols=0) {
	if(mxIsDouble(mxM) and mxIsSparse(mxM))
		return mxSparse_to_smat(mxM, R);
	else if(mxIsDouble(mxM) and !mxIsSparse(mxM)) {
		if(rows!=0 and cols !=0)
			return mxCoo_to_smat(mxM, R, rows, cols);
		else
			return mxDense_to_smat(mxM, R);
	}
}

/*
blocks_t mxSparse_to_blocks(const mxArray *M, int num_blocks, blocks_t &R) {
	R = blocks_t(num_blocks);
	unsigned long rows, cols, nnz;
	mwIndex *ir, *jc;
	double *v;
	ir = mxGetIr(M); jc = mxGetJc(M); v = mxGetPr(M);
	rows = mxGetM(M); cols = mxGetN(M); nnz = jc[cols];
	R.from_matlab(rows, cols, nnz);
	for(unsigned long c = 0; c < cols; c++) {
		for(unsigned long idx = jc[c]; idx < jc[c+1]; ++idx){
			R.insert_rate(idx, rate_t(ir[idx], c, v[idx]));
			++R.nnz_row[ir[idx]];
			++R.nnz_col[c];
		}
	}
	R.compressed_space(); // Need to call sort later.
	sort(R.allrates.begin(), R.allrates.end(), RateComp(&R));
	return R;
}

// convert matab dense matrix to column fmt
int mxDense_to_matCol(const mxArray *mxM, mat_t &M) {
	unsigned long rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M = mat_t(cols, vec_t(rows,0));
	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; ++r)
			M[c][r] = val[idx++];
	return 0;
}

int matCol_to_mxDense(const mat_t &M, mxArray *mxM) {
	unsigned long cols = M.size(), rows = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matCol_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; r++)
			val[idx++] = M[c][r];
	return 0;
}

// convert matab dense matrix to row fmt
int mxDense_to_matRow(const mxArray *mxM, mat_t &M) {
	unsigned long rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M = mat_t(rows, vec_t(cols,0));
	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; ++r)
			M[r][c] = val[idx++];
	return 0;
}

int matRow_to_mxDense(const mat_t &M, mxArray *mxM) {
	unsigned long rows = M.size(), cols = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matRow_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; r++)
			val[idx++] = M[r][c];
	return 0;
}

*/

bool check_identity(const mxArray *mxM) {
	size_t m = mxGetM(mxM), n = mxGetN(mxM);
	if(!mxIsDouble(mxM) or m!=n) return false;
	double *val = mxGetPr(mxM);
	if(mxIsSparse(mxM)) {
		mwIndex *ir = mxGetIr(mxM), *jc = mxGetJc(mxM);
		size_t nnz = jc[n];
		if(nnz != n) return false;
		for(size_t i = 0; i < n; i++)
			if(jc[i] != i or ir[i] != i or val[i] != 1.0)
				return false;
		return true;
	} else {
		size_t idx = 0;
		for(size_t i = 0; i < m; i++) {
			for(size_t j = 0; j < m; j++) {
				if(i!=j and val[idx]!=0)
					return false;
				if(i==j and val[idx]!=1.0)
					return false;
				idx++;
			}
		}
	}
	return true;
}
int get_matrix_type(const mxArray *mxM) {
	if(mxIsSparse(mxM)) {
		return imf_prob_t::Sparse;
	} else {
		return imf_prob_t::Dense;
	}
}

// if nrhs == 4 or 5 => arr_mf_train(Y, testY, lag_idx, lag_val, 'options')
// if nrhs == 6 or 7 => arr_mf_train(Y, testY, lag_idx, lag_val, W, H, 'options')
int run_arr_mf_train(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], arr_mf_param_t &param) {
	const mxArray *mxY=prhs[0], *mxtestY=prhs[1], *mxlag_idx=prhs[2], *mxlag_val=prhs[3], *mxinitW=NULL, *mxinitH=NULL;
	size_t m=mxGetM(mxY), n=mxGetN(mxY), k=param.k;
	size_t lag_size=mxGetN(mxlag_val);
	double *lag_val = mxGetPr(mxlag_val);
	size_t *lag_idx = MALLOC(size_t, lag_size);
	double *tmp_idx = mxGetPr(mxlag_idx);
	for(size_t d = 0; d < lag_size; d++)
		lag_idx[d] = (size_t) tmp_idx[d];

	double *W, *H;
	double *rmse=mxGetPr(plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL));
	double *walltime=mxGetPr(plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL));
	//double *cputime=mxGetPr(plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL));

	smat_t Y, testY;

	if(nrhs == 6 || nrhs == 7) {
		mxinitW = prhs[4];
		mxinitH = prhs[5];
		m = mxGetN(mxinitW);
		n = mxGetN(mxinitH);
		mxArray_to_smat(mxY, Y, m, n);
		mxArray_to_smat(mxtestY, testY, Y.rows, Y.cols);

		if( mxGetN(mxinitW) != m || mxGetN(mxinitH) != n || mxGetM(mxinitW) != mxGetM(mxinitH)) {
			printf("Error: Dimensions do not match!\n");
			return -1;
		}
		if(mxGetM(mxinitW) != k) {
			k = param.k = mxGetM(mxinitW);
			printf("Warning: Change param.k to %ld to match W0 and H0\n", k);
		}
		W = mxGetPr(plhs[0] = mxCreateDoubleMatrix(k,m,mxREAL));
		H = mxGetPr(plhs[1] = mxCreateDoubleMatrix(k,n,mxREAL));
		double *initW = mxGetPr(mxinitW), *initH = mxGetPr(mxinitH);
		for(size_t i = 0; i < k*m; i++) W[i] = initW[i];
		for(size_t i = 0; i < k*n; i++) H[i] = initH[i];
	} else {
		mxArray_to_smat(mxY, Y, m, n);
		mxArray_to_smat(mxtestY, testY, Y.rows, Y.cols);
		W = mxGetPr(plhs[0] = mxCreateDoubleMatrix(k,m,mxREAL));
		H = mxGetPr(plhs[1] = mxCreateDoubleMatrix(k,n,mxREAL));
		srand48(1);
		for(size_t i = 0; i < k*m; i++) W[i] = drand48();
		for(size_t i = 0; i < k*n; i++) H[i] = drand48();
	}


	if(mxGetM(mxlag_val) != k) {
		printf("Error: Dimensions do not match from lag_val (k * lag_size) !\n");
		return -1;
	}


	*walltime = omp_get_wtime();
	arr_mf_prob_t prob(&Y, lag_size, lag_idx, lag_val, k);
	arr_mf_train(&prob, &param, W, H, (testY.nnz)? &testY : NULL, rmse);
	*walltime = omp_get_wtime() - *walltime;
	if(lag_idx) free(lag_idx);
	return 0;
}


// Interface function of matlab
// now assume prhs[0]: A, prhs[1]: W, prhs[0]
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	arr_mf_param_t param;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	// Transform the input Matrix to libsvm format
	if(nrhs >= 4 && nrhs <= 7)
	{
		if (//isDoubleSparse(prhs[0])==false ||
				// isDoubleSparse(prhs[1])==false ||
				// isDoubleDense(prhs[1])==false  ||
				// isDoubleDense(prhs[2])==false ||
				// isDoubleDense(prhs[3])==false) {
				mxIsDouble(prhs[2]) == false ||
				mxIsDouble(prhs[3]) == false) {
			fake_answer(plhs);
			return;
		}
		if(!mxIsDouble(prhs[0]) or !mxIsDouble(prhs[1])) {
			mexPrintf("Error: matrix must be double\n");
			fake_answer(plhs);
			return;
		}

		param = parse_command_line(nrhs, prhs);
		switch (param.solver_type){
			case ARR_LS_FULL:
			case ARR_LS_MISSING:
				run_arr_mf_train(nlhs, plhs, nrhs, prhs, param);
				break;
			default:
				fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
				exit_with_help();
				fake_answer(plhs);
				break;
		}
	} else {
		exit_with_help();
		fake_answer(plhs);
	}

}


