function [opt_Z, f] = linprog_test(Z_init, A_org, delta_l, delta_g, R)
%   rng(4);
%   n = 54;
%   delta_l = 5; delta_g = 500;
%   A_org = randi(2, n, n) - 1;
%   A_org(1:(n+1):end) = 0;
%   A_org = min(A_org, A_org');
%   R = rand(n, n);
%   Z_init = A_org;
  
  n = size(A_org, 1);

  
%   generate inequality matrix 
  V = -2*A_org + 1;
  A_local_budget = zeros(n, n^2);
  b_local_budget = zeros(n, 1);
  for i=1:n
    A_local_budget(:, n*(i-1)+1:n*i) = diag(V(:, i));
    b_local_budget(i) = delta_l - sum(A_org(:, i));
  end
  A_global_budget = V(:)';
  b_global_budget = delta_g - sum(A_org, 'all');
  A = [A_local_budget; A_global_budget];
  b = [b_local_budget; b_global_budget];
  
%   generate equality matrix
  eq_len = n*(n-1)/2;
  Aeq = zeros(eq_len, n^2);
  beq = zeros(eq_len, 1);
  count = 0;
  for i=1:n
    for j=i+1:n
      count = count + 1;
      idx = sub2ind([n,n], i, j);
      idx1 = sub2ind([n,n], j, i);
      Aeq(count, idx) = 1;
      Aeq(count, idx1) = -1;
    end
  end
  
%   generate lower/upper bound
  lb = zeros(size(Z_init));
  ub = ones(size(Z_init));
  ub = ub - diag(diag(ub));   % diagonal be always 0 
  lb = lb(:);
  ub = ub(:);
%   fmincon_options = optimoptions(@fmincon,'Algorithm','interior-point', ...
%           'SpecifyObjectiveGradient',true, 'MaxFunctionEvaluations', 1000, ...
%           'optimalityTolerance',1e-3, 'Display','iter');
%   funObj = @(x)objective(x, R(:));
%   [opt_Z, f] = fmincon(funObj, Z_init, A, b, Aeq, beq, lb, ub, [], fmincon_options);
  linprog_options = optimoptions(@linprog, 'Algorithm','dual-simplex', ...
          'MaxIterations', 1000, 'optimalityTolerance',1e-6, 'Display','None');
  [opt_Z, f] = linprog(R(:), A, b, Aeq, beq, lb, ub, linprog_options);
  opt_Z = reshape(opt_Z, n, n);
%   opt_Z = reshape(opt_Z, n, n);
end

function [f, g] = objective(Z, flat_R)
  f = dot(Z(:), flat_R);
  g = flat_R;
end
