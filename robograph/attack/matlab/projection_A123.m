function proj_Z = projection_A123(Z, A_org, delta_l, delta_g)
  n = size(A_org, 1);
  Z = reshape(Z, n, n);
  proj_Z = Z;
  for i=1:50
    Z_begin = proj_Z;
    proj_Z = projection_A3(Z_begin);
    proj_Z = projection_coA1(delete_diagonal(proj_Z), delete_diagonal(A_org), delta_l);
    proj_Z = fill_diagonal(proj_Z);
    proj_Z = projection_coA2(delete_diagonal(proj_Z), delete_diagonal(A_org), delta_g);
    proj_Z = fill_diagonal(proj_Z);

    chanes = max(abs(proj_Z-Z_begin), [], 'all');
    if chanes < 1e-10
      break
    end
  end
  proj_Z = proj_Z(:);


  if sum((proj_Z-Z(:)).^2)/2 > 400
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
    lb = zeros(size(A_org));
    ub = ones(size(A_org));
    ub = ub - diag(diag(ub));   % diagonal be always 0
    lb = lb(:);
    ub = ub(:);

    quad_options = optimoptions('quadprog', 'MaxIterations', 1000, 'Display','None' );
    [x, fval] = quadprog(eye(n^2),-Z(:),A,b,Aeq,beq,lb,ub,[],quad_options);
    fprintf('alter = %f, lcqp = %f\n', sum((proj_Z-Z(:)).^2)/2, sum((x-Z(:)).^2)/2);
  end
end


function Z = delete_diagonal(X)
  Xtemp = X';
  Z = reshape(Xtemp(~eye(size(Xtemp))), size(X, 2)-1, [])';
end

function Z = fill_diagonal(X)
  r = size(X, 1);
  Z = [tril(X,-1) zeros(r, 1)] + [zeros(r,1) triu(X)];
end


function proj_Z = projection_A3(Z)
  proj_Z = (Z + transpose(Z))/2;
end