clc();
clear();
load('data');
componentDesc       = CDLib(X,y,'T',5,'n',round(numel(y)/5*4),'class',["knn","svm"]);
options             = optimoptions('ga');
options.Display     = 'iter';
options.UseParallel = 1;
ga(@(varargin)componentDesc(varargin{:}),5,[],[],[],[],zeros(1,5)+1e-6,inf(1,5),[],[],options)
