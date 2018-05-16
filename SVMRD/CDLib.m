classdef CDLib < matlab.mixin.internal.indexing.Paren
	properties
		varargin cell
	end
	methods
		function this = CDLib(X,y,varargin)
			varargin      = struct(varargin{:});
			this.varargin = [X,y,cellfun(@(field)varargin.(field),["T","n","class"],'un',0)];
		end
		function [p,daptc,daccu,accur,Z] = parenReference(this,sigma)
			[X,y,T,n,~] = this.varargin{:};
			[y,C,Z]     = deal(y(1:n),1,[]);
			D = sum((permute(X,[1,3,2])-permute(X,[3,1,2])).^2,3).^(1/2);
			v = {0*y-1,[],[],y.',0,0*y,0*y+C,[],optimoptions('quadprog','Display','off')};
			for t = 1:T
				K = sqrt(2*pi)*sigma(t)*normpdf(D,0,sigma(t));
				H = y*y.'.*K(1:n,1:n);
				a = quadprog(H,v{:});
				Z = [Z,((a.*y).'*K(1:n,:)/((a.*y).'*K(1:n,1:n)*(a.*y))^(1/2)).'];%#ok
				G = K-Z(:,t)*Z(:,t).';
				D = (diag(G)+diag(G).'-G-G.').^(1/2);
			end
			accur = this.accuracy(Z);
			[daccu,daptc] = deal(cellfun(@(accur)diff(accur,[],2),accur,'un',0));
			if any([daptc{:}]<=0)
				daptc = cellfun(@(daptc)daptc.*(daptc<=0),daptc,'un',0);
			end
			p = -100*sum([daptc{:}]);
		end
		function accur = accuracy(this,Z)
			[X,y,T,n,~] = this.varargin{:};
			accur = cellfun(@(f)[
				mean(predict(feval(f,X(1:n, : ),y(1:n)),X(n+1:end, : ))==y(n+1:end))...
				arrayfun(@(t)...
				mean(predict(feval(f,Z(1:n,1:t),y(1:n)),Z(n+1:end,1:t))==y(n+1:end))...
				,1:T)
				],'fitc'+string(this.varargin{end}),'un',0);
		end
		function p = disp(this,parameters,varargin)
			[~,~,T,~,class] = this.varargin{:};
			if nargin==1
				builtin('disp',this);
			else
				[p,daptc,daccu,accur,Z] = this(parameters);
				fprintf(1,'-----------------------------------------------------\n');
				fprintf(1,'Parameters::%s\n',replace(mat2str(parameters),' ',','));
				fprintf(1,'  Accuracy::%s\n',['Origin ',sprintf('     %d ',1:T)]);
				fprintf(1, '       %s::%s\n',accStr(class,accur));
				fprintf(1,' dAccuracy::%s\n',sprintf('     %d ',1:T));
				fprintf(1, '       %s::%s\n',accStr(class,daccu));
				fprintf(1,'dA-patched::%s\n',sprintf('     %d ',1:T));
				fprintf(1, '       %s::%s\n',accStr(class,daptc));
				fprintf(1,'   Perform::%s\n',num2str(-p,'%05.2f%%'));
				fprintf(1,'-----------------------------------------------------\n');
				plot(this,Z,varargin{:});
			end
			function s = accStr(class,accuracy)
				s = [upper(class);cellfun(@(accuracy)sprintf('%05.2f%% ',100*accuracy),accuracy,'un',0)];
			end
		end
		function plot(this,Z,varargin)
			[~,y,~,n,~] = this.varargin{:};
			cla();
			hold('on');
			class = unique(y);
			plot3(Z((1:end).'<=n&y(:)==class(1),1),Z((1:end).'<=n&y(:)==class(1),2),Z((1:end).'<=n&y(:)==class(1),3),varargin{:});
			plot3(Z((1:end).'<=n&y(:)==class(2),1),Z((1:end).'<=n&y(:)==class(2),2),Z((1:end).'<=n&y(:)==class(2),3),varargin{:});
			plot3(Z((1:end).'> n&y(:)==class(1),1),Z((1:end).'> n&y(:)==class(1),2),Z((1:end).'> n&y(:)==class(1),3),varargin{:});
			plot3(Z((1:end).'> n&y(:)==class(2),1),Z((1:end).'> n&y(:)==class(2),2),Z((1:end).'> n&y(:)==class(2),3),varargin{:});
			xlim(minmax(Z(:,1).'));
			ylim(minmax(Z(:,2).'));
			zlim(minmax(Z(:,3).'));
			axis('equal');
			xlabel('x');
			ylabel('y');
			zlabel('z');
			drawnow();
		end
	end
end