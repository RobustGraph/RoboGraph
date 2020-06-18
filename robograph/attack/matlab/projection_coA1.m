
function proj_Z = projection_coA1(Z, A_org, delta_l)
  if all(Z>=0, 'all') && all(Z<=1, 'all') && all(sum(abs(Z-A_org), 2) <= delta_l)
    proj_Z = Z;
  else
    proj_Z = zeros(size(Z));
    for i = 1:size(Z, 1)
      proj_Z(i, :) = projection_coA2(Z(i, :), A_org(i, :), delta_l);
    end
  end
end