function [opt_Z, f] = linprog1_test(A, b, Aeq, beq, lb, ub, R)
  linprog_options = optimoptions(@linprog, 'Algorithm','dual-simplex', ...
          'MaxIterations', 1000, 'optimalityTolerance',1e-6, 'Display','iter');
  [opt_Z, f] = linprog(R, A, b, Aeq, beq, lb, ub, linprog_options);
end

