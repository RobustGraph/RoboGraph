function [opt_Z, f] = lbfgsb_test(Z_init, A_org, delta_l, delta_g, R)
  rng(4);
  n = 50;
  delta_l = 5; delta_g = 500;
  A_org = randi(2, n, n) - 1;
  A_org(1:(n+1):end) = 0;
  A_org = min(A_org, A_org');
  R = rand(n, n);
  Z_init = A_org;

  funObj = @(x)objective(x, R);
  funProj = @(x)projection_A123(x, A_org, delta_l, delta_g);
  options = [];
  options.verbose = 2;
  options.maxIter = 100;
  options.SPGoptTol = 1e-8;
  [opt_Z, f, ~] = minConf_PQN(funObj, Z_init(:), funProj, options);
  opt_Z = reshape(opt_Z, size(R));
end

function [f, g] = objective(Z, R)
  flat_R = R(:);
  f = dot(Z, flat_R);
  g = flat_R;
end




