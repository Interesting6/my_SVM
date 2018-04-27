clc();
clf();
clear();
view(3);
hold('on');
load('data');
options = optimoptions('ga');
options.Display = 'diagnose';
options.MutationFcn = @mutationuniform;
options.UseParallel = 1;
obj([1.5478    4.4547   19.3012   19.3249   13.6885],X,y);
%ga(@(sigma)obj(sigma,X,y),5,[],[],[],[],zeros(1,5)+1e-6,[],[],options);
function d = obj(sigma,X,y)
	sigma%#ok
	[C,n,l,T]         = deal(1,500,size(X,1),size(sigma,2));
	[D,K,H,G]         = deal(zeros(l,l,T));
	[f,Aeq,beq,lb,ub] = deal(0*y-1,y.',0,0*y,0*y+C);
	[alpha,Z]   = deal(zeros(n,T),zeros(l,T));
	for t = 1:T
		if t==1
			D(:,:,t) = sum((permute(X,[1,3,2])-permute(X,[3,1,2])).^2,3).^(1/2);
		else
			D(:,:,t) = (diag(G(:,:,t-1))+diag(G(:,:,t-1)).'-G(:,:,t-1)-G(:,:,t-1).').^(1/2);
		end
		% sigma(t) = std(reshape(D(:,:,t),[numel(D(:,:,t)),1]));
		if t==1
			K(:,:,t) = sqrt(2*pi)*sigma(t)*normpdf(D(:,:,t),0,sigma(t)).*(     sum(X.^2,2)*     sum(X.^2,2).').^(1/2);
		else
			K(:,:,t) = sqrt(2*pi)*sigma(t)*normpdf(D(:,:,t),0,sigma(t)).*(diag(G(:,:,t-1))*diag(G(:,:,t-1)).').^(1/2);
		end
		H(:,:,t)   = y*y.'.*K(:,:,t);
		alpha(:,t) = quadprog(H(1:n,1:n,t),f(1:n),[],[],Aeq(1:n),beq,lb(1:n),ub(1:n),[],optimoptions('quadprog','Display','off'));
		Z(:,t)     = ((alpha(:,t).*y(1:n)).'*K(1:n,:,t)/((alpha(:,t).*y(1:n)).'*K(1:n,1:n,t)*(alpha(:,t).*y(1:n)))^(1/2)).';
		G(:,:,t)   = K(:,:,t)-Z(:,t)*Z(:,t).';
	end
	%%{
	cla();
	plot3(Z((1:end)<=n&y.'== 1,1),Z((1:end)<=n&y.'== 1,2),Z((1:end)<=n&y.'== 1,3),'o');
	plot3(Z((1:end)<=n&y.'==-1,1),Z((1:end)<=n&y.'==-1,2),Z((1:end)<=n&y.'==-1,3),'o');
	plot3(Z((1:end)> n&y.'== 1,1),Z((1:end)> n&y.'== 1,2),Z((1:end)> n&y.'== 1,3),'o');
	plot3(Z((1:end)> n&y.'==-1,1),Z((1:end)> n&y.'==-1,2),Z((1:end)> n&y.'==-1,3),'o');
	drawnow();
	%}
	M = fitcknn(X(1:n,:),y(1:n),'NumNeighbors',1);
	p = sum(M.predict(X(n+1:end,:))==y(n+1:end));
	for t = 1:T
		try
			M      = fitcknn(Z(1:n,1:t),y(1:n),'NumNeighbors',1);
		catch ME
			disp(ME.message);
			d = inf;
			return
		end
		p(t+1) = sum(M.predict(Z(n+1:end,1:t))==y(n+1:end));
	end
	d = diff(p);
	if any(d<0)
		d(d>0)  =  0;
	elseif any(d==0)
		d(d==0) = -1;
	end
	d%#ok
	d = -sum(d);
	Mdl = fitcsvm(Z(1:n,1:T),y(1:n));
	sum(Mdl.predict(Z(n+1:end,1:T))==y(n+1:end))
end
