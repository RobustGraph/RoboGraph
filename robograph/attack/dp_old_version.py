import numpy as np

def local_solver(A_org, XWU, nG, delta_l):
	"""
	solver of equation 6 of the paper when activation is identity, max_margin loss and average pooling	
	
    #Parameters:

    #Returns:
    #   a: matrix of local problem
	# 	A
	"""
	a = np.zeros((nG+1, delta_l+1)) 	# matrix a for described in equation 6 
	A_opt = np.zeros((nG+1, delta_l+1, nG))
	
	indices = np.argsort(XWU)
	for i in range(1, nG+1):	# looping each row of A
		A_i = A_org[i-1,:]
		A_i_edges = np.sum(A_i)
		canonical = np.zeros(nG); canonical[i-1] = 1.0 
		chunk_edges, chunk_no_edges = [], []
		for x in indices:
			if x == i-1: continue	# excluding self edge
			if A_i[x] == 1:
				chunk_edges.append((XWU[x], x))
			else:
				chunk_no_edges.append((XWU[x], x))
		chunk_edges.reverse()

		for j in range(0,delta_l+1):	# looping each possible local constraint
			if j == 0:
				A_opt[i, j] = A_i
				a[i,j] = np.dot(A_i + canonical, XWU)/(A_i_edges+1)
			else:
				min_f = float('inf')
				temp = []
				for k in range(j+1):  # looping different combinations of adding/removing
					add_edges, remove_edges = k, j-k
					if A_i_edges+add_edges > nG-1 or A_i_edges-remove_edges < 0:
						continue

					f = np.dot(A_i+canonical, XWU)
					# adding k edges from chunk of A_i=0 by ascent order
					add_edge_idx = []
					if add_edges > 0:
						edge_val, add_edge_idx = zip(*chunk_no_edges[0:add_edges])
						f += np.sum(edge_val)

					# removing j-k edges from chunk of A_i=1 by descent order
					remove_edge_idx = []
					if remove_edges > 0:
						edge_val, remove_edge_idx = zip(*chunk_edges[0:remove_edges])
						f -= np.sum(edge_val)

					final_f = f/(A_i_edges+add_edges-remove_edges + 1)
					temp.append( (final_f, add_edge_idx, remove_edge_idx) )
					if final_f < min_f:
						min_f = final_f
						sol = (final_f, add_edge_idx, remove_edge_idx)
				
				final_f, add_edge_idx, remove_edge_idx = sol
				A_new_i = A_i.copy()
				A_new_i[list(add_edge_idx)] = 1
				A_new_i[list(remove_edge_idx)] = 0
				A_opt[i, j] = A_new_i
				f_1 = final_f
				# f_2 = np.dot(A_new_i+canonical,XWU)/(A_i_edges+len(add_edge_idx)-len(remove_edge_idx) + 1)
				a[i,j] = f_1
	return a, A_opt


def local_solver_linear_term(A_org, XWU, nG, delta_l, linear_matrix):
	"""
	solver of equation 8&11 of the paper when activation is identity, max_margin loss and average pooling	
	"""
	a = np.zeros((nG+1, delta_l+1)) 	# matrix a for described in equation 6 
	A_opt = np.zeros((nG+1, delta_l+1, nG))
	# complexity: nG^2 delta_l log(nG) + nG*delta_l^2
	# 50^2*10*5 + 50*100 = 130000 / 10^7 = 1/100 
	L = linear_matrix
	for i in range(1, nG+1):	# looping each row of A
		A_i = A_org[i-1,:]
		A_i_edges = np.sum(A_i)
		L_i = L[:, i-1]
		canonical = np.zeros(nG); canonical[i-1] = 1.0 
		# precomuting chunk_edges and chunk_no_edges
		# need to loop possible values of (1'A_i+1) since it is in the numerator 
		# ( A_i@V + I_i@V+ A_i @ L_i*(1'A_i + 1) )/(1'A_i + 1)
		chunk_edges_mtx, chunk_no_edges_mtx = [None]*(nG+1), [None]*(nG+1)
		max_edges = min(A_i_edges + delta_l + 1, nG) 
		min_edges = max(A_i_edges - delta_l + 1, 1)
		for x in range(min_edges, max_edges+1):  # looping all possible (1'A_i + 1)
			V_L = XWU + L_i.T*x
			indices = np.argsort(V_L)
			edge_temp, no_edge_temp = [], []
			for y in indices:
				if y == i-1: continue	# excluding self edge
				if A_i[y] == 1:
					edge_temp.append((V_L[y], y))
				else:
					no_edge_temp.append((V_L[y], y))
			edge_temp.reverse()

			# edge_temp = [[V_L[y], y] for y in reversed(indices) if A_i[y]==1 and y!=(i-1)]
			# no_edge_temp = [[V_L[y], y] for y in indices if A_i[y]==0 and y!=(i-1)]
			# for s1 in range(1, len(edge_temp)):
			# 	edge_temp[s1][0] += edge_temp[s1-1][0]
			# for s1 in range(1, len(no_edge_temp)):
			# 	no_edge_temp[s1][0] += no_edge_temp[s1-1][0]
			
			chunk_edges_mtx[x] = edge_temp
			chunk_no_edges_mtx[x] = no_edge_temp
		
		# chunk_edges_mtx, chunk_no_edges_mtx = [], []
		# for x in range(nG+1):
		# 	V_L = XWU + L_i.T*x
		# 	indices = np.argsort(V_L)
		# 	edge_temp, no_edge_temp = [], []
		# 	for y in indices:
		# 		if y == i-1: continue	# excluding self edge
		# 		if A_i[y] == 1:
		# 			edge_temp.append((V_L[y], y))
		# 		else:
		# 			no_edge_temp.append((V_L[y], y))
		# 	edge_temp.reverse()
		# 	chunk_edges_mtx.append(edge_temp)
		# 	chunk_no_edges_mtx.append(no_edge_temp)

		A_V_i = np.dot(A_i + canonical, XWU)
		A_L_i = np.dot(A_i, L_i)
		for j in range(0,delta_l+1):	# looping each possible local constraint
			if j == 0:
				A_opt[i, j] = A_i
				a[i,j] = A_V_i/(A_i_edges+1) + A_L_i
			else:
				min_f = float('inf')
				temp = []
				for k in range(j+1):  # looping different combinations of adding/removing
					add_edges, remove_edges = k, j-k
					if A_i_edges+add_edges > nG-1 or A_i_edges-remove_edges < 0:
						continue

					new_edges = A_i_edges+add_edges-remove_edges + 1
					f = A_V_i + A_L_i*new_edges

					# adding k edges from chunk of A_i=0 by ascent order
					add_edge_idx = []
					if add_edges > 0:
						chunk_no_edges = chunk_no_edges_mtx[new_edges]
						edge_val, add_edge_idx = zip(*chunk_no_edges[0:add_edges])
						f += np.sum(edge_val)
						# edge_val, add_edge_idx = zip(*chunk_no_edges[0:add_edges])
						# f += edge_val[-1]

					# removing j-k edges from chunk of A_i=1 by descent order
					remove_edge_idx = []
					if remove_edges > 0:
						chunk_edges = chunk_edges_mtx[new_edges]
						edge_val, remove_edge_idx = zip(*chunk_edges[0:remove_edges])
						f -= np.sum(edge_val)
						# edge_val, remove_edge_idx = zip(*chunk_edges[0:remove_edges])
						# f -= edge_val[-1]

					final_f = f/new_edges
					temp.append( (final_f, add_edge_idx, remove_edge_idx) )
					if final_f < min_f:
						min_f = final_f
						sol = (final_f, add_edge_idx, remove_edge_idx)
				
				final_f, add_edge_idx, remove_edge_idx = sol
				A_new_i = A_i.copy()
				A_new_i[list(add_edge_idx)] = 1
				A_new_i[list(remove_edge_idx)] = 0
				A_opt[i, j] = A_new_i
				f_1 = final_f
				# f_2 = np.dot(A_new_i+canonical,XWU)/(A_i_edges+len(add_edge_idx)-len(remove_edge_idx) + 1)
				a[i,j] = f_1
	return a, A_opt


def dp_solver(A_org, XWU, nG, delta_l, delta_g, linear_term=[]):
	"""
	DP for min_{A_G^{1+2}} F_c(A)  (Algorithm 1) 

    #Parameters:

    #Returns:
    #   A_pert: optimal attack adjacency matrix
	"""
	# precomputing a matrix
	if len(linear_term) > 0:
		a, A_opt = local_solver_linear_term(A_org, XWU, nG, delta_l, linear_term)
	else:
		a, A_opt = local_solver(A_org, XWU, nG, delta_l)

	A_pert = np.zeros((nG,nG))
	s = np.ones((nG+1, min(nG*delta_l, delta_g)+1))*float('inf')
	s[0,0] = 0
	c = np.zeros(nG+1,dtype=int)
	# nG*min(nG*delta_l, delta_g)*delta_l = 50*500*10 = 250000
	for t in range(1, nG+1):
		c[t] = int(min(c[t-1] + delta_l, delta_g))
		for j in range(0,c[t]+1):
			m = float('inf')
			for k in range(0, delta_l+1):
				if j-k>=0 and j-k<=c[t-1]:
					s[t,j] = min(s[t-1,j-k]+a[t,k], m)
					m = s[t,j]
	J = np.array([0]*(nG+1))
	J[nG] = np.argmin(s[nG,1:])+1
	# print(s)
	# print(J[nG])
	K = np.zeros(nG+1,dtype=int)
	for t in range(nG,0,-1):
		temp = np.ones(delta_l+1)*float('inf')
		for k in range(0,delta_l+1):
			if (J[t] - k)>=0 and (J[t]-k)<= c[t-1]:
				temp[k] = s[t-1,J[t]-k] + a[t,k]
		K[t] = int(np.argmin(temp))
		J[t-1] = J[t] - K[t]
		A_pert[t-1,:] = A_opt[t,K[t]]

	unpert_val = s[nG, 0]
	opt_val = s[nG, J[nG]]
	return (unpert_val, opt_val, A_pert)

if __name__ == "__main__":
	np.random.seed(1)

	delta_l,delta_g = 2, 7	 #Setting the delta_l and delta_g values
	nG = 4	 # Setting theNumber of Nodes

	h = 7	 # dimension of hidden representation
	XW = np.random.randn(nG, h)
	U = np.random.randn(h)
	XWU = XW @ U

	A_org = np.random.randint(2, size = (nG,nG))	# Random Adjacency Matrix 
	for i in range(nG):
		A_org[i, i] = 0 # Adjacency Matrix must be zero diagonal

	lam = np.random.randn(nG, nG)*5
	linear_term = np.transpose(lam) - lam
	sol = dp_solver(A_org, XWU, nG, delta_l, delta_g, linear_term)
	unpert_val, opt_val, A_pert = sol
	print(A_org, unpert_val)
	print(A_pert, opt_val)
	print(np.sum(abs(A_org-A_pert), axis=1))

	





