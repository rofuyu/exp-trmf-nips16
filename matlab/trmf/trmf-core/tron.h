#ifndef _TRON_H
#define _TRON_H

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
	virtual void init(){}
};

class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, int max_iter = 100, int max_cg_iter = 20, bool pure_cg=false, double cg_eps=0.1);
	~TRON();

	void tron(double *w, bool set_w_to_zero = true);
	void set_print_string(void (*i_print) (const char *buf));
	void set_eps(double eps) {this->eps = eps;}

private:
	int trcg(double delta, double *g, double *s, double *r, double *cg_rnorm);
	double norm_inf(int n, double *x);

	double eps, cg_eps;
	bool pure_cg;
	int max_iter;
	int max_cg_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
	// local variables for tron
	double *s, *r, *w_new, *g;
	// local variables for trcg
	double *d, *Hd;

};
#endif
