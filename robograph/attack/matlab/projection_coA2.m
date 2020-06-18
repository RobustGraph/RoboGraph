
function proj_Z = projection_coA2(Z, A_org, delta_g)
  if all(Z>=0, 'all') && all(Z<=1, 'all') && sum(abs(Z-A_org), 'all') <= delta_g
    proj_Z = Z;
  else
    param = [];
    param.maxIter = 10;     % max number of iterations

    ub = inf;  lb = 0;
    fun = @(x)dual_obj(x, Z, A_org, delta_g);
    [opt_lamb, fval, iter, numCall, msg] = lbfgsb(0.1, lb, ub, fun, [], [], param);
    proj_Z = Z - opt_lamb*(-2*A_org+1);
    proj_Z(proj_Z>1) = 1;
    proj_Z(proj_Z<0) = 0;
  end
end


function [f, g] = dual_obj(lamb, Z, A_org, delta_g)
  opt_z = Z - lamb*(-2*A_org+1);
  opt_z(opt_z>1) = 1;
  opt_z(opt_z<0) = 0;
  s = sum(abs(opt_z - A_org));
  f = sum((opt_z-Z).^2)/2 + lamb*( s - delta_g);
  f = -f;
  g = delta_g - s;
end