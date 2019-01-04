function make(cc, cxx)
if nargin < 2
	cxx = 'g++';
	cc = 'gcc';
end
% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
try
	Type = ver;
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		% arr_mf_train
		cxx_flag = '"-fopenmp -ffast-math -pipe -O3 -DNDEBUG"';
		inc_dir = '/u/rofuyu/bin/octave-dev/usr/include/octave-3.8.1/octave/';
		ld_dir = '/u/rofuyu/.local/lib';
		blas_options = '-lblas -llapack';
		%env_var = sprintf('CXX=%s CC=%s CXXFLAGS=''-fopenmp -ffast-math -pipe -O3 -DNDEBUG -I/u/rofuyu/bin/octave-dev/usr/include/octave-3.8.1/octave/ '' DL_LDFLAGS=''-shared -Wl,-Bsymbolic -L/u/rofuyu/.local/lib '' DL_LD=%s LD_CXX=%s', cxx,cc, cxx,cxx);
		env_var = sprintf('CXX=%s CC=%s CXXFLAGS=%s DL_LD=%s LD_CXX=%s', cxx, cc, cxx_flag, cxx, cxx);
		disp(env_var)
		system(sprintf('%s make -C ../ lib ', env_var));
		dep_obj = '../imf.o ../dbilinear.o ../smat.o ../dmat.o ../tron.o ../zlib_util.o ../zlib/libz.a ';
		cmd = sprintf('%s mkoctfile -v -lgomp %s -I%s -L%s --mex arr_mf_train.cpp %s', env_var, blas_options, inc_dir, ld_dir, dep_obj);
		disp(cmd)
		system(cmd);
		
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
	else
		system('make -C ../ lib ');

		% arr_mf_train
		mex -largeArrayDims -lmwlapack -lmwblas CFLAGS="\$CFLAGS -fopenmp " LDFLAGS="\$LDFLAGS -fopenmp  -Wall " COMPFLAGS="\$COMPFLAGS -openmp" -cxx arr_mf_train.cpp  ../imf.o ../dbilinear.o  ../smat.o ../dmat.o ../tron.o ../zlib_util.o ../zlib/libz.a 

	end
catch
	fprintf('If make.m failes, please check README about detailed instructions.\n');
end
