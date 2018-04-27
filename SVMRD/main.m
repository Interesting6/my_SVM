clear
clf
view(3)
hold on
load data
X = 1000000000000000000000.*X;
C = 1;
n = 500;
for t = 1:3
	if t==1
		D(:,:,t) = sum((permute(X,[1,3,2])-permute(X,[3,1,2])).^2,3).^(1/2);
	else
		D(:,:,t) = (diag(Khat(:,:,t-1))+diag(Khat(:,:,t-1)).'-Khat(:,:,t-1)-Khat(:,:,t-1).').^(1/2);
	end
	K(:,:,t)    = normpdf(D(:,:,t),0,std(reshape(D(:,:,t),[numel(D(:,:,t)),1])));
	H(:,:,t)    = y*y.'.*K(:,:,t);
	f(:,t)      = 0*y-1;
	Aeq(t,:)    = y.';
	beq(t)      = 0;
	lb(:,t)     = 0*y;
	ub(:,t)     = C+0*y;
	alpha(:,t)  = quadprog(H(1:n,1:n,t),f(1:n,t),[],[],Aeq(t,1:n),beq(t),lb(1:n,t),ub(1:n,t));
	Z(:,t)      = ((alpha(:,t).*y(1:n)).'*K(1:n,:,t)/((alpha(:,t).*y(1:n)).'*K(1:n,1:n,t)*(alpha(:,t).*y(1:n)))^(1/2)).';
	Khat(:,:,t) = K(:,:,t)-Z(:,t)*Z(:,t).';
end
plot3(Z((1:end)<=n&y.'== 1,1),Z((1:end)<=n&y.'== 1,2),Z((1:end)<=n&y.'== 1,3),'o');
plot3(Z((1:end)<=n&y.'==-1,1),Z((1:end)<=n&y.'==-1,2),Z((1:end)<=n&y.'==-1,3),'o');
plot3(Z((1:end)> n&y.'== 1,1),Z((1:end)> n&y.'== 1,2),Z((1:end)> n&y.'== 1,3),'o');
plot3(Z((1:end)> n&y.'==-1,1),Z((1:end)> n&y.'==-1,2),Z((1:end)> n&y.'==-1,3),'o');