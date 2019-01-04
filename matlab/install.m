function install(lang)
	if ismac()
		cc='gcc-5'; cxx='g++-5';
	else
		cc='gcc'; cxx='g++';
	end
	if nargin == 0
		Type = ver;
		% This part is for OCTAVE
		if(strcmp(Type(1).Name, 'Octave') == 1)
			lang = 'octave';
		else
			lang = 'matlab';
		end
	end

	root = pwd();
	p = sprintf('%s/trmf/trmf-core/%s',root,lang);
	cd(p); make(cc,cxx); cd(root);
	addpath('trmf');
	%addpath(p);
	%addpath(sprintf('%s/SpaRSA',root));
end
