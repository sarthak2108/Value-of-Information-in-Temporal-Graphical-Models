#########################################################################################
# CODE FOR OPTIMIZING VALUE OF INFORMATION (VOI) IN TEMPORAL GRAPHICAL MODELS IN
# THE CONDITIONAL PLANNING SETTING OVER AN INFINITE TIME HORIZON.
# ALSO INCLUDES IMPLEMENTATIONS OF BASELINE TECHNIQUES, SUCH AS VoIDP
# (KRAUSE AND GUESTRIN, [2005]), AND THE UNIFORM SPACING HEURISTIC.
#########################################################################################

#########################################################################################
# EXTERNAL.
#########################################################################################
from copy import deepcopy
from itertools import permutations
from math import log
from math import inf
from math import pow
from math import ceil
import numpy
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

#########################################################################################
# MDP NODES USED BY OUR ALGORITHM.
#########################################################################################
class MDP:
	def __init__(self, domain, X, O, gamma, reward, o, k, m, B):
		self.X = X
		self.O = O
		self.gamma = gamma
		self.local_reward = reward
		self.o = o
		self.k = k
		self.m = m
		self.B = B
		self.old_reward = reward
		self.new_reward = reward
		self.select_state = [None for i in range(domain)]
		self.select_prob = [-1 for i in range(domain)]
		self.skip_state = None
		self.choice = -1

#########################################################################################
# MDP NODES USED BY VoIDP (KRAUSE AND GUESTRIN, 2005)
#########################################################################################	
class finite_MDP:
	def __init__(self, domain):
		self.select_state = [None for i in range(domain)]
		self.select_prob = [0.0 for i in range(domain)]
		self.skip_state = None
		self.Ob = None
		self.reward = 1
		self.choice = -1

#########################################################################################
# GLOBAL DICTIONARIES.
#########################################################################################	
prob = dict()
local_reward = dict()
mdp_states = dict()

#########################################################################################
# CUDA SOURCE MODULE.
#########################################################################################
mod = SourceModule("""
__global__ void update(float *old_value, float *new_value, float *local, int *select, float *prob, int *skip, float *pars)
{
	const int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < int(pars[2])){
		if((select[i * int(pars[0])] != -1) && (skip[i] != -1)){
			float sel = 0;
			float sk = 0;
			for(int j = 0; j < int(pars[0]); j++){
				sel += prob[i * int(pars[0]) + j] * old_value[select[i * int(pars[0]) + j]];
			}
			sk = old_value[skip[i]];
			if(sel > sk){
				new_value[i] = local[i] + pars[1] * sel;
			}
			else{
				new_value[i] = local[i] + pars[1] * sk;
			}
		}
		else if(select[i * int(pars[0])] == -1){
			new_value[i] = local[i] + pars[1] * old_value[skip[i]];
		}
		else{
			new_value[i] = local[i];
			for(int j = 0; j < int(pars[0]); j++){
				new_value[i] += pars[1] * prob[i * int(pars[0]) + j] * old_value[select[i * int(pars[0]) + j]];
			}
		}
	}
	else{
		new_value[i] = old_value[i];
	}
}
""")

#########################################################################################
# UNCOMMENT THE FOLLOWING TWO MODELS AS NECESSARY, BUT NOT SIMULTAENEOUSLY.
#########################################################################################
#########################################################################################
# CHAIN MODEL FOR THE TEMPERATURE TASK (BASED ON DATA AVAILABLE AT
# http://db.csail.mit.edu/labdata/labdata.html).
#########################################################################################
prior = [0.00031575623618566466,0.016103568045468898,0.3514366908746448,0.5775181559835807,0.05462582886011999]
transition = [[0.2,0.2,0.2,0.2,0.2],
[0.01818181818181818,0.7090909090909091,0.12727272727272726,0.12727272727272726,0.01818181818181818],
[0.0008968609865470852,0.011659192825112108,0.8726457399103139,0.10852017937219731,0.006278026905829596],
[0.0005461496450027307,0.0005461496450027307,0.0726379027853632,0.9016930638995084,0.024576734025122882],
[0.005714285714285714,0.005714285714285714,0.005714285714285714,0.28,0.7028571428571428]]
#########################################################################################
# HMM FOR THE PSYCHIATRY TASK (PARAMETERS TAKEN FROM HUANG, 2016).
#########################################################################################
'''prior = [.1312, .3077, .3323, .1423, .0865]
emission = [[0.734, 0.196, 0.049, 0.017, 0.004],
[0.429, 0.275, 0.155, 0.116, 0.025],
[0.123, 0.354, 0.261, 0.215, 0.046],
[0.074, 0.226, 0.228, 0.278, 0.194],
[0.024, 0.098, 0.195, 0.341, 0.341]]
transition = [[0.7638, 0.224, 0.012, 0.0001, 0.0001],
[0.109, 0.6659, 0.198, 0.028, 0.0001],
[0.002, 0.121, 0.717, 0.152, 0.009],
[0.0001, 0.004, 0.195, 0.6299, 0.172],
[0.0001, 0.0001, 0.002, 0.175, 0.8228]]'''

#########################################################################################
# UNCOMMENT WHEN SOLVING THE CHAIN MODEL.
#########################################################################################
def next(X):
	return (1, X[1] + 1)
#########################################################################################
# UNCOMMENT WHEN SOLVING THE HMM.
#########################################################################################		
'''def next(X):
	if X[0] == 0:
		return (1, X[1])
	else:
		return (0, X[1] + 1)'''

#########################################################################################
# GENERATING THE MDP CORRESPONDING TO OUR ALGORITHM.
#########################################################################################		
def plan_MDP(domain, k, m, B, max_B, gamma):
	#########################################################################################
	# WE EMPLOY THE DICTIONARY reg_ID TO KEEP TRACK OF AN ORDERING OF MDP STATES.
	#########################################################################################
	MDP_states = dict()
	reg_ID = dict()
	pen = 0.0
	s, id = MDP(domain, (-1, -1), (((-1, -1), -1),), gamma, 0.0, 0, k - 1, m, B), 0
	WL = [s]
	while len(WL) != 0:
		s = WL.pop(0)
		if (s.X, s.O, s.o, s.k, s.m, s.B) not in MDP_states:
			if s.X[0] == 0 or s.X[0] == -1:
				#########################################################################################
				# NON-SEPARATOR VARIABLE.
				#########################################################################################
				new_X = next(s.X)
				if new_X[0] == 0:
					#########################################################################################
					# THE NEXT VARIABLE IS ALSO A NON-SEPARATOR VARIABLE, AND HENCE OBSERVED BY DEFAULT.
					#########################################################################################
					for i in range(domain):
						new_O = list(s.O)
						if new_O == [((-1, -1), -1)]:
							new_O = [(new_X, i)]
						else:
							new_O.append((new_X, i))
						new_s = MDP(domain, new_X, tuple(new_O), gamma, 0.0, s.o, s.k, s.m, s.B)
						WL.append(new_s)
						s.select_state[i] = new_s
						s.select_prob[i] = prob[(new_X, i), s.O]
						s.choice = 1
				else:
					#########################################################################################
					# THE NEXT VARIABLE IS A SEPARATOR VARIABLE.
					#########################################################################################
					if s.k != 0 or s.o != 0:
						#########################################################################################
						# THE CONSTRAINTS ALLOW US TO SKIP THIS VARIABLE.
						#########################################################################################
						new_O = list(s.O)
						new_s = MDP(domain, new_X, tuple(new_O), gamma, local_reward[new_X, tuple(new_O)], s.o, s.k, s.m, s.B)
						WL.append(new_s)
						s.skip_state = new_s
					if (s.B > s.m) or (s.B >= s.m and s.o == 0):
						#########################################################################################
						# THE CONSTRAINTS ALLOW US TO OBSERVE THIS VARIABLE.
						#########################################################################################
						new_Xs = (1, 0)
						for i in range(domain):
							new_O = [(new_Xs, i)]
							new_s = MDP(domain, new_Xs, tuple(new_O), gamma, 0.0 - pen, 1, s.k, s.m - (1 - s.o), s.B - 1)
							WL.append(new_s)
							s.select_state[i] = new_s
							s.select_prob[i] = prob[(new_X, i), s.O]
					'''if s.k == 0 and s.o == 0:
						new_Xs = (1, 0)
						for i in range(domain):
							new_O = [(new_Xs, i)]
							new_s = MDP(domain, new_Xs, new_O, gamma, 1, s.k, s.m - (1 - s.o), s.B - 1)
							WL.append(new_s)
							s.select_state[i] = new_s
							s.select_prob[i] = prob((new_Xs, i), s.O)
							s.choice = 1
					elif s.B <= s.m or s.B == 0:
						new_O = deepcopy(s.O)
						new_s = MDP(domain, new_X, new_O, gamma, s.o, s.k, s.m, s.B)
						WL.append(new_s)
						s.skip_state = new_s
						s.choice = 0
					else:
						new_Xs = (1, 0)
						for i in range(domain):
							new_O = [(new_Xs, i)]
							new_s = MDP(domain, new_Xs, new_O, gamma, 1, s.k, s.m - (1 - s.o), s.B - 1)
							WL.append(new_s)
							s.select_state[i] = new_s
							s.select_prob[i] = prob((new_Xs, i), s.O)
						new_O = deepcopy(s.O)
						new_s = MDP(domain, new_X, new_O, gamma, s.o, s.k, s.m, s.B)
						WL.append(new_s)
						s.skip_state = new_s'''
			else:
				#########################################################################################
				# SEPARATOR VARIABLE.
				#########################################################################################
				new_X = next(s.X)
				if new_X[0] == 0:
					#########################################################################################
					# THE NEXT VARIABLE IS A NON-SEPARATOR VARIABLE, AND HENCE OBSERVED BY DEFAULT.
					#########################################################################################
					for i in range(domain):
						new_O = list(s.O)
						new_O.append((new_X, i))
						if s.k == 0 and s.m == 0:
							new_s = MDP(domain, new_X, tuple(new_O), gamma, 0.0, 0, k - 1, m, min(s.B + B, max_B))
						elif s.k == 0 and s.m != 0:
							new_s = MDP(domain, new_X, tuple(new_O), gamma, 0.0, 0, k - 1, s.m, s.B)
						else:
							new_s = MDP(domain, new_X, tuple(new_O), gamma, 0.0, s.o, s.k - 1, s.m, s.B)
						WL.append(new_s)
						s.select_state[i] = new_s
						s.select_prob[i] = prob[(new_X, i), s.O]
						s.choice = 1
				else:
					#########################################################################################
					# THE NEXT VARIABLE IS A SEPARATOR VARIABLE.
					#########################################################################################
					if s.k != 1 or s.o != 0:
						#########################################################################################
						# THE CONSTRAINTS ALLOW US TO SKIP THIS VARIABLE.
						#########################################################################################
						new_O = list(s.O)
						if s.k == 0 and s.m == 0:
							new_s = MDP(domain, new_X, tuple(new_O), gamma, local_reward[new_X, tuple(new_O)], 0, k - 1, m, min(s.B + B, max_B))
						elif s.k == 0 and s.m != 0:	
							new_s = MDP(domain, new_X, tuple(new_O), gamma, local_reward[new_X, tuple(new_O)], 0, k - 1, s.m, s.B)
						else:
							new_s = MDP(domain, new_X, tuple(new_O), gamma, local_reward[new_X, tuple(new_O)], s.o, s.k - 1, s.m, s.B)
						WL.append(new_s)
						s.skip_state = new_s
					if (s.B > s.m) or (s.B >= s.m and s.o == 0) or (s.B >= s.m and s.k == 0):
						#########################################################################################
						# THE CONSTRAINTS ALLOW US TO OBSERVE THIS VARIABLE.
						#########################################################################################
						new_Xs = (1, 0)
						for i in range(domain):
							new_O = [(new_Xs, i)]
							if s.k == 0 and s.m == 0:
								new_s = MDP(domain, new_Xs, tuple(new_O), gamma, 0.0 - pen, 1, k - 1, m - 1, min(s.B - 1 + B, max_B))
							elif s.k == 0 and s.m != 0:	
								new_s = MDP(domain, new_Xs, tuple(new_O), gamma, 0.0 - pen, 1, k - 1, s.m - 1, s.B - 1)
							else:
								new_s = MDP(domain, new_Xs, tuple(new_O), gamma, 0.0 - pen, 1, s.k - 1, s.m - (1 - s.o), s.B - 1)
							WL.append(new_s)
							s.select_state[i] = new_s
							s.select_prob[i] = prob[(new_X, i), s.O]
					'''if s.k == 1 and s.o == 0:
						new_Xs = (1, 0)
						for i in range(domain):
							new_O = [(new_Xs, i)]
							s.select_prob[i] = prob[(new_X, i), s.O]
							new_s = MDP(domain, new_Xs, new_O, gamma, 1, s.k - 1, s.m - 1, s.B - 1)
							WL.append(new_s)
							s.select_state[i] = new_s
							s.select_prob[i] = prob((new_Xs, i), s.O)
							s.choice = 1
					elif s.B <= s.m or s.B == 0:
						new_O = deepcopy(s.O)
						if s.k == 0 and s.m == 0:
							new_s = MDP(domain, new_X, new_O, gamma, 0, k - 1, m, min(s.B + B, max_B))
						elif s.k == 0 and s.m != 0:	
							new_s = MDP(domain, new_X, new_O, gamma, 0, k - 1, s.m, s.B)
						else:
							new_s = MDP(domain, new_X, new_O, gamma, s.o, s.k - 1, s.m, s.B)
						WL.append(new_s)
						s.skip_state = new_s
						s.choice = 0
					else:
						new_Xs = (1, 0)
						for i in range(domain):
							new_O = [(new_Xs, i)]
							if s.k == 0 and s.m == 0:
								new_s = MDP(domain, new_Xs, new_O, gamma, 1, k - 1, m - 1, min(s.B - 1 + B, max_B))
							elif s.k == 0 and s.m != 0:	
								new_s = MDP(domain, new_Xs, new_O, gamma, 1, k - 1, s.m - 1, s.B - 1)
							else:
								new_s = MDP(domain, new_Xs, new_O, gamma, 1, s.k - 1, s.m - (1 - s.o), s.B - 1)
							WL.append(new_s)
							s.select_state[i] = new_s
							s.select_prob[i] = prob((new_Xs, i), s.O)
						new_O = deepcopy(s.O)
						if s.k == 0 and s.m == 0:
							new_s = MDP(domain, new_X, new_O, gamma, 0, k - 1, m, min(s.B + B, max_B))
						elif s.k == 0 and s.m != 0:	
							new_s = MDP(domain, new_X, new_O, gamma, 0, k - 1, s.m, s.B)
						else:
							new_s = MDP(domain, new_X, new_O, gamma, s.o, s.k - 1, s.m, s.B)
						WL.append(new_s)
						s.skip_state = new_s'''
			MDP_states[(s.X, s.O, s.o, s.k, s.m, s.B)] = s
			reg_ID[(s.X, s.O, s.o, s.k, s.m, s.B)] = id
			id += 1
	#########################################################################################
	# ENSURE ALL SUCCESSOR STATES ARE IN THE DICTIONARY.
	#########################################################################################
	for state in MDP_states:
		if MDP_states[state].select_state[0] != None:
			for i in range(domain):
				outsider = MDP_states[state].select_state[i]
				MDP_states[state].select_state[i] = MDP_states[(outsider.X, outsider.O, outsider.o, outsider.k, outsider.m, outsider.B)]
		if MDP_states[state].skip_state != None:
			outsider = MDP_states[state].skip_state
			MDP_states[state].skip_state = MDP_states[(outsider.X, outsider.O, outsider.o, outsider.k, outsider.m, outsider.B)]
	return 	MDP_states, reg_ID

#########################################################################################
# VALUE ITERATION.
#########################################################################################	
def find_plan(MDP_states, domain, epsilon, k):
	if k == inf:
		#########################################################################################
		# WE ITERATE UNTIL THE BELLMAN RESIDUE BECOMES LESS THAN epsilon.
		#########################################################################################
		j = 0
		diff = inf
		while diff >= epsilon:
			diff = -inf
			for state in MDP_states:
				MDP_states[state].old_reward = MDP_states[state].new_reward
			for state in MDP_states:
				if MDP_states[state].select_state[0] == None:
					select_reward = -inf
				else:
					select_reward = 0.0
					for i in range(domain):
						select_reward += MDP_states[state].select_prob[i] * MDP_states[state].select_state[i].old_reward
				if MDP_states[state].skip_state == None:
					skip_reward = -inf
				else:
					skip_reward = MDP_states[state].skip_state.old_reward
				MDP_states[state].new_reward = MDP_states[state].local_reward + MDP_states[state].gamma * max(select_reward, skip_reward)
				if select_reward > skip_reward:
					MDP_states[state].choice = 1
				else:
					MDP_states[state].choice = 0
				if diff < abs(MDP_states[state].new_reward - MDP_states[state].old_reward):
					diff = abs(MDP_states[state].new_reward - MDP_states[state].old_reward)
			j += 1
	else:
		#########################################################################################
		# WE ITERATE k TIMES.
		#########################################################################################
		j = k + 1
		diff = -inf
		for i in range(k + 1):
			if diff == 0:
				break
			for state in MDP_states:
				MDP_states[state].old_reward = MDP_states[state].new_reward
			diff = -inf
			for state in MDP_states:
				select_reward = 0.0
				if MDP_states[state].select_state[0] == None:
					select_reward = -inf
				else:
					for l in range(domain):
						select_reward += MDP_states[state].select_prob[l] * MDP_states[state].select_state[l].old_reward
				if MDP_states[state].skip_state == None:
					skip_reward = -inf
				else:
					skip_reward = MDP_states[state].skip_state.old_reward
				MDP_states[state].new_reward = MDP_states[state].local_reward + MDP_states[state].gamma * max(select_reward, skip_reward)
				if select_reward > skip_reward:
					MDP_states[state].choice = 1
				else:
					MDP_states[state].choice = 0
					MDP_states[state].new_reward = MDP_states[state].local_reward + MDP_states[state].gamma * skip_reward
				if diff < abs(MDP_states[state].new_reward - MDP_states[state].old_reward):
					diff = abs(MDP_states[state].new_reward - MDP_states[state].old_reward)
					
	print(j)
	return MDP_states, diff
	
#########################################################################################
# VALUE ITERATION USING CUDA CORES.
#########################################################################################
update = mod.get_function("update")

def find_plan_CUDA(MDP_states, ID, domain, gamma, k):
	#########################################################################################
	# THE MDP_states DICTIONARY IS CAST IN FORM OF numpy ARRAYS USING DICTIONARY ID.
	#########################################################################################
	local = [0.0 for i in range(len(ID))]
	select = [0 for i in range(len(ID) * domain)]
	prob = [0.0 for i in range(len(ID) * domain)]
	skip = [0 for i in range(len(ID))]
	for state in MDP_states:
		local[int(ID[state])] = MDP_states[state].local_reward
		if MDP_states[state].select_state[0] == None:
			for i in range(domain):
				select[int(ID[state]) * domain + i] = -1
				prob[int(ID[state]) * domain + i] = -1
		else:
			for i in range(domain):
				s = MDP_states[state].select_state[i]
				select[int(ID[state]) * domain + i] = ID[(s.X, s.O, s.o, s.k, s.m, s.B)]
				prob[int(ID[state]) * domain + i] = MDP_states[state].select_prob[i]
		if MDP_states[state].skip_state == None:
			skip[int(ID[state])] = -1
		else:
			s = MDP_states[state].skip_state
			skip[int(ID[state])] = ID[(s.X, s.O, s.o, s.k, s.m, s.B)]
	#########################################################################################
	# PAD TO AVOID pycuda.drv.LogicError.
	#########################################################################################
	auxl = numpy.asarray(local).astype(numpy.float32)
	local = numpy.zeros((len(ID) + 1024 - (len(ID) % 1024),), dtype = numpy.float32)
	local[:auxl.shape[0]] = auxl
	select = numpy.asarray(select).astype(numpy.int)
	prob = numpy.asarray(prob).astype(numpy.float32)
	skip = numpy.asarray(skip).astype(numpy.int)
	new = numpy.zeros_like(local)
	old = numpy.zeros_like(local)
	pars = [domain, gamma, len(ID)]
	pars = numpy.asarray(pars).astype(numpy.float32)
	#########################################################################################
	# CREATE GPU COPIES OF numpy ARRAYS.
	#########################################################################################
	local_gpu = gpuarray.to_gpu(local)
	select_gpu = gpuarray.to_gpu(select)
	prob_gpu = gpuarray.to_gpu(prob)
	skip_gpu = gpuarray.to_gpu(skip)
	new_gpu = gpuarray.to_gpu(new)
	old_gpu = gpuarray.to_gpu(old)
	pars_gpu = gpuarray.to_gpu(pars)
	#########################################################################################
	# ITERATE A PREDETERMINED NUMBER OF TIMES, k. BLOCK AND GRID SIZES MAY NEED TO BE ALTERED
	# DEPENDING ON COMPUTE CAPABILITY VERSION. VERSION USED: 2.1.
	#########################################################################################
	for i in range(k):
		update(old_gpu, new_gpu, local_gpu, select_gpu, prob_gpu, skip_gpu, pars_gpu, block = (1024, 1, 1), grid = (int(local.size / 1024), 1, 1))
		old_gpu, new_gpu = new_gpu, old_gpu
	return old_gpu[0]
	
#########################################################################################
# MDP CORRESPONDING TO THE UNIFORM SPACING HEURISTIC.
#########################################################################################
def uniform_MDP(domain, frac, gamma):
	#########################################################################################
	# frac DEFINES THE SPACING.
	#########################################################################################
	#########################################################################################
	# WE EMPLOY THE DICTIONARY uni_ID TO KEEP TRACK OF AN ORDERING OF MDP STATES.
	#########################################################################################
	uniform_states = dict()
	uni_ID = dict()
	pen = 0.0
	s, id = MDP(domain, (-1, -1), (((-1, -1), -1),), gamma, 0.0, (frac[1], frac[1]), 0, 0, 0), 0
	WL =[s]
	while len(WL) != 0:
		s = WL.pop(0)
		if (s.X, s.O, s.o, 0, 0, 0) not in uniform_states:
			new_X = next(s.X)
			if new_X[0] == 0:
				#########################################################################################
				# THE NEXT VARIABLE IS A NON_SEPARATOR VARIABLE, AND HENCE OBSERVED BY DEFAULT.
				#########################################################################################
				for i in range(domain):
					new_O = list(s.O)
					if new_O == [((-1, -1), -1)]:
						new_O = [(new_X, i)]
					else:
						new_O.append((new_X, i))
					new_s = MDP(domain, new_X, tuple(new_O), gamma, 0.0, s.o, 0, 0, 0)
					WL.append(new_s)
					s.select_state[i] = new_s
					s.select_prob[i] = prob[(new_X, i), s.O]
			else:
				#########################################################################################
				# THE NEXT VARIABLE IS A SEPARATOR VARIABLE.
				#########################################################################################
				if s.o[0] - frac[1] < frac[1]:
					new_Xs = (1, 0)
					for i in range(domain):
						new_O = [(new_Xs, i)]
						new_s = MDP(domain, new_Xs, tuple(new_O), gamma, 0.0 - pen, (s.o[0] - frac[1] + frac[0], frac[1]), 0, 0, 0)
						WL.append(new_s)
						s.select_state[i] = new_s
						s.select_prob[i] = prob[(new_X, i), s.O]
				else:
					new_O = list(s.O)
					new_s = MDP(domain, new_X, tuple(new_O), gamma, local_reward[new_X, tuple(new_O)], (s.o[0] - frac[1], frac[1]), 0, 0, 0)
					WL.append(new_s)
					s.skip_state = new_s
			uniform_states[(s.X, s.O, s.o, 0, 0, 0)] = s
			uni_ID[(s.X, s.O, s.o, 0, 0, 0)] = id
			id += 1
	#########################################################################################
	# ENSURE ALL SUCCESSOR STATES ARE IN THE DICTIONARY.
	#########################################################################################
	for state in uniform_states:
		if uniform_states[state].select_state[0] != None:
			for i in range(domain):
				outsider = uniform_states[state].select_state[i]
				uniform_states[state].select_state[i] = uniform_states[(outsider.X, outsider.O, outsider.o, 0, 0, 0)]
		if uniform_states[state].skip_state != None:
			outsider = uniform_states[state].skip_state
			uniform_states[state].skip_state = uniform_states[(outsider.X, outsider.O, outsider.o, 0, 0, 0)]
	return 	uniform_states, uni_ID				

#########################################################################################
# MORE EFFICIENT IMPLEMENTATION OF VoIDP (GHOSH AND RAMAKRISHNAN, 2017).
#########################################################################################	
def e_plan_MDP(select, Xi, Vi, Ob, n, B, domain, gamma, first):
	pen = 0.0
	s = finite_MDP(domain)
	if select == True or Xi == -1:
		new_Ob = (Xi, Vi)
		new_first = first
		if Xi == -1:
			new_B = B
			loc_reward = 0.0
		else:
			new_B = B - 1
			if first == -1:
				new_first = Xi
			loc_reward = 0.0 - pen
	else:
		new_Ob = Ob
		new_B = B
		new_first = first
		loc_reward = finite_local_reward(Xi, new_Ob)
	s.Ob = new_Ob
	if new_B < 0:
		return None, -inf, -1
	if (Xi, new_Ob, new_B, new_first) in mdp_states:
		return mdp_states[(Xi, new_Ob, new_B, new_first)][0], mdp_states[(Xi, new_Ob, new_B, new_first)][1], mdp_states[(Xi, new_Ob, new_B, new_first)][2]
	if Xi < n:
		new_Xi = Xi + 1
		select_reward = 0.0
		for i in range(domain):
			s.select_state[i], reward, select_first = e_plan_MDP(True, new_Xi, i, new_Ob, n, new_B, domain, gamma, new_first)
			s.select_prob[i] = finite_prob((new_Xi, i), new_Ob)
			select_reward += s.select_prob[i] * reward
		s.skip_state, skip_reward, skip_first = e_plan_MDP(False, new_Xi, -1, new_Ob, n, new_B, domain, gamma, new_first)
		s.reward = loc_reward + gamma * max(select_reward, skip_reward)
		if select_reward > skip_reward:
			s.choice = 1
			newer_first = select_first
		else:
			s.choice = 0
			newer_first = skip_first
	else:
		s.choice = -1
		s.reward = loc_reward
		newer_first = new_first
	mdp_states[(Xi, new_Ob, new_B, new_first)] = (s, s.reward, newer_first, s.choice, Xi, new_Ob)
	return s, s.reward, newer_first
	
'''def compute_reward(domain, s, n, first, first_r):
	if s in aux_mdp_states:
		return aux_mdp_states[s]
	else:
		cur_reward = 0.0
		if s.choice == 1:
			for i in range(domain):
				cur_reward += s.select_prob[i] * compute_reward(domain, s.select_state[i], n, first, first_r)
		elif s.choice == 0:
			cur_reward += compute_reward(domain, s.skip_state, n, first, first_r)
		else:
			j = 1
			for i in range(n - s.Ob[0] + 1, n - s.Ob[0] + first + 1):
				cur_reward += pow(gamma, j) * local_reward[(1, i), (((1, 0), s.Ob[1]),)]
				j += 1
			temp = 0.0
			for i in range(domain):
				temp += prob[(((1, n - s.Ob[0] + first + 1), i), (((1, 0), s.Ob[1]),))] * first_r[i]
			cur_reward = pow(gamma, j) * temp
		aux_mdp_states[s] = cur_reward
		return cur_reward'''
		
#########################################################################################
# FUNCTIONS FOR PERFORMING PROBABILISTIC INFERENCES IN CHAIN MODELS.
#########################################################################################
def prob_infers(domain, k):
	for i in range(domain):
		prob[(((1, 0), i), (((-1, -1), -1),))] = prior[i]
	for i in range(1, k):
		for v1 in range(domain):
			temp = 0.0
			for v2 in range(domain):
				temp += prob[(((1, i - 1), v2), (((-1, -1), -1),))] * transition[v2][v1]
			prob[(((1, i), v1), (((-1, -1), -1),))] = temp
	for i in range(domain):
		for j in range(domain):
			prob[(((1, 1), i), (((1, 0), j),))] = transition[j][i]
	for i in range(2, 2*k):
		for v1 in range(domain):
			for v2 in range(domain):
				temp = 0.0
				for v3 in range(domain):
					temp += prob[(((1, i - 1), v3), (((1, 0), v2),))] * transition[v3][v1]
				prob[(((1, i), v1), (((1, 0), v2),))] = temp
				
def finite_prob(Xi, Ob):
	if Ob[0] == -1:
		return prob[(((1, Xi[0]), Xi[1]), (((-1, -1), -1),))]
	else:
		return prob[(((1, Xi[0] - Ob[0]), Xi[1]), (((1, 0), Ob[1]),))]
		
#########################################################################################
# FUNCTIONS FOR COMPUTING LOCAL REWARDS IN CHAIN MODELS.
#########################################################################################
#########################################################################################
# RESIDUAL ENTROPY.
#########################################################################################
def loc_rewards(domain, k):
	for i in range(k):
		temp = 0.0
		for j in range(domain):
			temp += (prob[(((1, i), j), (((-1, -1), -1),))] * log(prob[(((1, i), j), (((-1, -1), -1),))]))
		local_reward[((1, i), (((-1, -1), -1),))] = temp
	for i in range(1, 2*k):
		for j in range(domain):
			temp = 0.0
			for l in range(domain):
				temp += (prob[(((1, i), l), (((1, 0), j),))] * log(prob[(((1, i), l), (((1, 0), j),))]))
			local_reward[((1, i), (((1, 0), j),))] = temp
			
def finite_local_reward(Xi, Ob):
	if Ob[0] == -1:
		return local_reward[(1, Xi), (((-1, -1), -1),)]
	else:
		return local_reward[(1, Xi - Ob[0]), (((1, 0), Ob[1]),)]
		
#########################################################################################
# MARGIN OF CONFIDENCE.
#########################################################################################		
def loc_rewards_mc(domain, k):
	for i in range(k):
		max1, max2 = -inf, -inf
		for j in range(domain):
			if prob[(((1, i), j), (((-1, -1), -1),))] > max1:
				max2 = max1
				max1 = prob[(((1, i), j), (((-1, -1), -1),))]
			elif prob[(((1, i), j), (((-1, -1), -1),))] > max2:
				max2 = prob[(((1, i), j), (((-1, -1), -1),))]
		local_reward[((1, i), (((-1, -1), -1),))] = max1 - max2
	for i in range(1, 2*k):
		for j in range(domain):
			max1, max2 = -inf, -inf
			for l in range(domain):
				if prob[(((1, i), l), (((1, 0), j),))] > max1:
					max2 = max1
					max1 = prob[(((1, i), l), (((1, 0), j),))]
				elif prob[(((1, i), l), (((1, 0), j),))] > max2:
					max2 = prob[(((1, i), l), (((1, 0), j),))]
			local_reward[((1, i), (((1, 0), j),))] = max1 - max2
			
#########################################################################################
# FUNCTION FOR PERFORMING PROBABILISTIC INFERENCES AND COMPUTING LOCAL REWARDS IN HMMS.
#########################################################################################		
def prob_infers_nc(domain, k):
	#########################################################################################
	# THE SEPARATOR VARIABLE HAS NOT BEEN OBSERVED.
	#########################################################################################
	l = []
	for i in range(domain):
		for j in range(2 * k):
			l.append(i)
	for i in range(domain):
		prob[(((1, 0), i), (((-1, -1), -1),))] = prior[i]
	for i in range(domain):
		temp = 0.0
		for j in range(domain):
			temp += emission[j][i] * prob[(((1, 0), j), (((-1, -1), -1),))]
		prob[(((0, 0), i), (((-1, -1), -1),))] = temp
	for j in range(domain):
		temp = 0.0
		for i in range(domain):
			prob[(((1, 0), i), (((0, 0), j),))] = prob[(((1, 0), i), (((-1, -1), -1),))] * emission[i][j] / prob[(((0, 0), j), (((-1, -1), -1),))]
			temp += prob[(((1, 0), i), (((0, 0), j),))] * log(prob[(((1, 0), i), (((0, 0), j),))])
		local_reward[((1, 0), (((0, 0), j),))] = temp
	for i in range(1, k):
		vs = []
		for v in permutations(l, i + 1):
			if v not in vs:
				vs.append(v)
		for v in vs:
			ls = []
			for j in range(i + 1):
				ls.append(((0, j), v[j]))
			for a in range(domain):
				temp = 0.0
				for b in range(domain):
					temp += transition[b][a] * prob[(((1, i - 1), b), tuple(ls[0:i]))]
				prob[(((1, i), a), tuple(ls[0:i]))] = temp
			for a in range(domain):
				temp = 0.0
				for b in range(domain):
					temp += emission[b][a] * prob[(((1, i), b), tuple(ls[0:i]))]
				prob[(((0, i), a), tuple(ls[0:i]))] = temp
			temp = 0.0
			for a in range(domain):
				prob[(((1, i), a), tuple(ls[0:i + 1]))] = emission[a][ls[i][1]] * prob[(((1, i), a), tuple(ls[0:i]))] / prob[(((0, i), ls[i][1]), tuple(ls[0:i]))]
				temp += prob[(((1, i), a), tuple(ls[0:i + 1]))] * log(prob[(((1, i), a), tuple(ls[0:i + 1]))])
			local_reward[((1, i), tuple(ls[0:i + 1]))] = temp
		del vs
	#########################################################################################
	# THE SEPARATOR VARIABLE HAS BEEN OBSERVED.
	#########################################################################################
	for i in range(domain):
		for j in range(domain):
			prob[(((1, 1), i), (((1, 0), j),))] = transition[j][i]
	for i in range(domain):
		for j in range(domain):
			temp = 0.0
			for a in range(domain):
				temp += emission[a][i] * prob[(((1, 1), a), (((1, 0), j),))]
			prob[(((0, 1), i), (((1, 0), j),))] = temp
	for i in range(domain):
		for j in range(domain):
			temp = 0.0
			for a in range(domain):
				prob[(((1, 1), a), (((1, 0), i), ((0, 1), j)))] = emission[a][j] * prob[(((1, 1), a), (((1, 0), i),))] / prob[(((0, 1), j), (((1, 0), i),))]
				temp += prob[(((1, 1), a), (((1, 0), i), ((0, 1), j)))] * log(prob[(((1, 1), a), (((1, 0), i), ((0, 1), j)))])
			local_reward[((1, 1), (((1, 0), i), ((0, 1), j)))] = temp
	for i in range(2, 2 * k):
		vs = []
		for v in permutations(l, i + 1):
			if v not in vs:
				vs.append(v)
		for v in vs:
			ls = [((1, 0), v[0])]
			for j in range(1, i + 1):
				ls.append(((0, j), v[j]))
			for a in range(domain):
				temp = 0.0
				for b in range(domain):
					temp += transition[b][a] * prob[(((1, i - 1), b), tuple(ls[0:i]))]
				prob[(((1, i), a), tuple(ls[0:i]))] = temp
			for a in range(domain):
				temp = 0.0
				for b in range(domain):
					temp += emission[b][a] * prob[(((1, i), b), tuple(ls[0:i]))]
				prob[(((0, i), a), tuple(ls[0:i]))] = temp
			temp = 0.0
			for a in range(domain):
				prob[(((1, i), a), tuple(ls[0:i + 1]))] = emission[a][ls[i][1]] * prob[(((1, i), a), tuple(ls[0:i]))] / prob[(((0, i), ls[i][1]), tuple(ls[0:i]))]
				temp += prob[(((1, i), a), tuple(ls[0:i + 1]))] * log(prob[(((1, i), a), tuple(ls[0:i + 1]))])
			local_reward[((1, i), tuple(ls[0:i + 1]))] = temp
		del vs

#########################################################################################
# MDP CORRESPONDING TO INFINITELY REPEATING THE FINITE-HORIZON PLAN DETERMINED BY VoIDP.
# THIS IS ONLY FOR CHAIN MODELS SINCE VoIDP IS ONLY APPLICABLE TO CHAINS.
#########################################################################################	
def greedy_MDP(domain, k, f, i):
	#########################################################################################
	# f is DETERMINED BY e_plan_MDP.
	#########################################################################################
	new_mdp_states = dict()
	pen = 0.0
	#########################################################################################
	# WE KEEP TRACK OF THE ACTION TO BE TAKEN UNDER EACH SCENARIO IN DICTIONARY
	# new_mdp_states.
	#########################################################################################
	for state in mdp_states:
		if state[1] == (-1, -1):
			if state[0] == -1:
				new_mdp_states[((-1, -1), (((-1, -1), -1),), -1, state[2])] = mdp_states[state][0].choice
			else:
				new_mdp_states[((1, state[0]), (((-1, -1), -1),), state[0], state[2])] = mdp_states[state][0].choice
		else:
			new_mdp_states[((1, state[0] - state[1][0]), (((1, 0), state[1][1]),), state[0], state[2])] = mdp_states[state][0].choice
	#########################################################################################
	# WE EMPLOY THE DICTIONARY reg_ID TO KEEP TRACK OF AN ORDERING OF MDP STATES.
	#########################################################################################
	greedy_states = dict()
	greedy_ID = dict()
	s, id = MDP(domain, (-1, -1), (((-1, -1), -1),), gamma, 0.0, -1, 0, 0, i), 0
	WL =[s]
	while len(WL) != 0:
		s = WL.pop(0)
		if (s.X, s.O, s.o, 0, 0, s.B) not in greedy_states:
			new_X = next(s.X)
			if s.o == f - 1 or s.o - k == f - 1:
				new_Xs = (1, 0)
				for j in range(domain):
					new_O = [(new_Xs, j)]
					if s.o + 1 == k:
						new_o = 0
						new_B = min(i, s.B + i)
					else:
						new_o = s.o + 1
						new_B = s.B
					new_s = MDP(domain, new_Xs, tuple(new_O), gamma, 0.0 - pen, new_o, 0, 0, new_B - 1)
					WL.append(new_s)
					s.select_state[j] = new_s
					s.select_prob[j] = prob[(new_X, j), s.O]
			elif s.o < f - 1:
				new_O = list(s.O)
				if s.o + 1 == k:
					new_o = 0
					new_B = min(i, s.B + i)
				else:
					new_o = s.o + 1
					new_B = s.B
				new_s = MDP(domain, new_X, tuple(new_O), gamma, local_reward[new_X, tuple(new_O)], new_o, 0, 0, new_B)
				WL.append(new_s)
				s.skip_state = new_s
			else:
				if new_mdp_states[(s.X, s.O, s.o, s.B)] == 1:
					new_Xs = (1, 0)
					for j in range(domain):
						new_O = [(new_Xs, j)]
						if s.o + 1 == k:
							new_o = 0
							new_B = min(i, s.B + i)
						else:
							new_o = s.o + 1
							new_B = s.B
						new_s = MDP(domain, new_Xs, tuple(new_O), gamma, 0.0 - pen, new_o, 0, 0, new_B - 1)
						WL.append(new_s)
						s.select_state[j] = new_s
						s.select_prob[j] = prob[(new_X, j), s.O]
				else:
					new_O = list(s.O)
					if s.o + 1 == k:
						new_o = 0
						new_B = min(i, s.B + i)
					else:
						new_o = s.o + 1
						new_B = s.B
					new_s = MDP(domain, new_X, tuple(new_O), gamma, local_reward[new_X, tuple(new_O)], new_o, 0, 0, new_B)
					WL.append(new_s)
					s.skip_state = new_s
			greedy_states[(s.X, s.O, s.o, 0, 0, s.B)] = s
			greedy_ID[(s.X, s.O, s.o, 0, 0, s.B)] = id
			id += 1
	#########################################################################################
	# ENSURE ALL SUCCESSOR STATES ARE IN THE DICTIONARY.
	#########################################################################################
	for state in greedy_states:
		if greedy_states[state].select_state[0] != None:
			for j in range(domain):
				outsider = greedy_states[state].select_state[j]
				greedy_states[state].select_state[j] = greedy_states[(outsider.X, outsider.O, outsider.o, 0, 0, outsider.B)]
		if greedy_states[state].skip_state != None:
			outsider = greedy_states[state].skip_state
			greedy_states[state].skip_state = greedy_states[(outsider.X, outsider.O, outsider.o, 0, 0, outsider.B)]
	return	greedy_states, greedy_ID

if __name__ == '__main__':
	#########################################################################################
	# DO NOT UNCOMMENT BOTH OF THE FOLLOWING SECTIONS SIMULTAENEOUSLY.
	#########################################################################################
	#########################################################################################
	# domain: SIZE OF DOMAIN OF VARIABLES IN THE MODEL.
	# k: THE SEPARATOR VARIABLE MUST BE OBSERVED AT LEAST ONCE IN k TIME STEPS.
	# m: NUMBER OF k-LENGTH INTERVALS AT WHICH BUDGET GETS REPLENISHED.
	# gamma: DISCOUNT FACTOR.
	# d: CONTROLS THE NUMBER OF ITERATIONS
	#########################################################################################
	#########################################################################################
	# UNCOMMENT THE NEXT SECTION TO RUN THE ALGORITHM FOR THE TEMPERATURE TASK (CHAIN MODEL).
	#########################################################################################
	domain, k, m, gamma, d = 5, 12, 1, 0.999, inf
	prob_infers(domain, k)
	loc_rewards(domain, k)
	for i in range(1, k + 1):
		#########################################################################################
		# i: INITIAL BUDGET. GETS REPLENISHED TO i EVERY k * m TIME STEPS.
		#########################################################################################
		s, r, f = e_plan_MDP(False, -1, -1, (-1, -1), k - 1, i, domain, gamma, -1)
		greedy_states, greedy_ID = greedy_MDP(domain, k, f, i)
		#greedy_states, diff1 = find_plan(greedy_states, domain, 0.000001, d)
		#inf_g = greedy_states[((-1, -1), (((-1, -1), -1),), -1, i)].old_reward
		inf_g = find_plan_CUDA(greedy_states, greedy_ID, domain, gamma, 30000)
		del greedy_states
		MDP_states, reg_ID = plan_MDP(domain, k, m, i, i, gamma)
		#MDP_states, diff2 = find_plan(MDP_states, domain, 0.000001, d)
		#inf_r = MDP_states[((-1, -1), (((-1, -1), -1),), 0, k - 1, m, i)].old_reward
		inf_r = find_plan_CUDA(MDP_states, reg_ID, domain, gamma, 30000)
		del MDP_states
		uniform_states, uni_ID = uniform_MDP(domain, (k , i), gamma)
		#uniform_states, diff3 = find_plan(uniform_states, domain, 0.000001, d)
		#inf_u = uniform_states[((-1, -1), (((-1, -1), -1),), 1)].old_reward
		inf_u = find_plan_CUDA(uniform_states, uni_ID, domain, gamma, 30000)
		del uniform_states
		print('[', inf_g, ',', inf_u, ',', inf_r, '],')
	#########################################################################################
	# UNCOMMENT THE NEXT SECTION TO RUN THE ALGORITHM FOR THE PSYCHIATRY TASK (HMM).
	#########################################################################################
	'''domain, k, m, gamma, d = 5, 2, 12, pow(0.99, 0.5), 12.5
	prob_infers_nc(domain, k)
	for i in range(23, (k * m)):
		#########################################################################################
		# i: INITIAL BUDGET. GETS REPLENISHED TO i EVERY k * m TIME STEPS.
		#########################################################################################
		MDP_states, reg_ID = plan_MDP(domain, k, m, i, i, gamma)
		#MDP_states, diff1 = find_plan(MDP_states, domain, 0.0001, int(k * m * d) - 1)
		#inf_r = MDP_states[((-1, -1), (((-1, -1), -1),), 0, k - 1, m, i)].old_reward
		inf_r = find_plan_CUDA(MDP_states, reg_ID, domain, gamma, 10000)
		del MDP_states
		uniform_states, uni_ID = uniform_MDP(domain, ((k * m), i), gamma)
		#uniform_states, diff2 = find_plan(uniform_states, domain, 0.0001, int(k * m * d) - 1)
		#inf_ru = uniform_states[((-1, -1), (((-1, -1), -1),), 1)].old_reward
		inf_u = find_plan_CUDA(uniform_states, uni_ID, domain, gamma, 10000)
		del uniform_states
		print('[', inf_r, ',', inf_u, '],')'''
