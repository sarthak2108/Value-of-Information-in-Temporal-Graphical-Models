#########################################################################################
# CODE FOR OPTIMIZING VALUE OF INFORMATION (VOI) IN CHAIN TEMPORAL GRAPHICAL MODELS IN
# THE SUBSET SELECTION SETTING.
# ALSO INCLUDES IMPLEMENTATIONS OF BASELINE TECHNIQUES, SUCH AS VOIDP
# (KRAUSE AND GUESTRIN, [2005]), GREEDY HEURISTIC AND THE UNIFORM SPACING HEURISTIC.
#########################################################################################

#########################################################################################
# EXTERNAL PROCEDURES
#########################################################################################
from math import log
from math import inf
from time import clock

#########################################################################################
# CHAIN MODEL FOR THE TEMPERATURE TASK (BASED ON DATA AVAILABLE AT
# http://db.csail.mit.edu/labdata/labdata.html).
# THE MODEL INVOLVES 24 TIME STEPS, EACH 60 MINUTES APART, STARTING FROM MIDNIGHT.
# DIVIDED INTO FOUR ZONES OF 6 HOURS EACH 0000 - 0060, 0006 - 0012, 0012 - 0018, 0018 -
# 0000.
#########################################################################################
t = [[[0.8983050847457628,0.05084745762711865,0.01694915254237288,0.01694915254237288,0.01694915254237288],
[0.0695970695970696,0.9194139194139194,0.003663003663003663,0.003663003663003663,0.003663003663003663],
[0.006134969325153374,0.10429447852760736,0.8773006134969326,0.006134969325153374,0.006134969325153374],
[0.034482758620689655,0.034482758620689655,0.034482758620689655,0.8620689655172413,0.034482758620689655],
[0.2,0.2,0.2,0.2,0.2]],
[[0.6065573770491803,0.21311475409836064,0.14754098360655737,0.01639344262295082,0.01639344262295082],
[0.005847953216374269,0.7076023391812866,0.25146198830409355,0.029239766081871343,0.005847953216374269],
[0.00425531914893617,0.00425531914893617,0.9148936170212766,0.07234042553191489,0.00425531914893617],
[0.017543859649122806,0.017543859649122806,0.3684210526315789,0.5789473684210527,0.017543859649122806],
[0.2,0.2,0.2,0.2,0.2]],
[[0.2,0.2,0.2,0.2,0.2],
[0.030303030303030304,0.7575757575757576,0.15151515151515152,0.030303030303030304,0.030303030303030304],
[0.002544529262086514,0.01272264631043257,0.9185750636132316,0.06361323155216285,0.002544529262086514],
[0.010752688172043012,0.010752688172043012,0.13978494623655913,0.8279569892473119,0.010752688172043012],
[0.2,0.2,0.2,0.2,0.2]],
[[0.2,0.2,0.2,0.2,0.2],
[0.03816793893129771,0.9083969465648855,0.03816793893129771,0.007633587786259542,0.007633587786259542],
[0.0029850746268656717,0.12835820895522387,0.8567164179104477,0.008955223880597015,0.0029850746268656717],
[0.018867924528301886,0.018867924528301886,0.2830188679245283,0.660377358490566,0.018867924528301886],
[0.2,0.2,0.2,0.2,0.2]]]
p = [0.054187192118226604,0.2788177339901478,0.554679802955665,0.11133004926108374,0.0009852216748768472]

#########################################################################################
# GLOBAL DICTIONARIES.
#########################################################################################
exp_local_rewards = dict()
mdp_states = dict()

#########################################################################################
# MDP NODES USED BY THE TECHNIQUES PRESENTED IN GHOSH AND RAMAKRISHNAN [2017].
#########################################################################################
class MDP:
	def __init__(self):
		self.select_state = None
		self.skip_state = None
		self.reward = 1
		self.choice = -1

#########################################################################################
# HELPER FUNCTIONS.
#########################################################################################		
def get_domain(X):
	if X == -1:
		return [-1]
	else:
		return [0, 1, 2, 3, 4]
	
def cond_prob(X1, v1, X2, v2):
	return t[int(X2) // 6][v2][v1]

def prior(X, v):
	return p[v]
	
#########################################################################################
# IMPLEMENTATION OF eSubsetMDP.
#########################################################################################
def e_subset_MDP(select, Xi, Ob, n, B):
	s = MDP()
	if select == True or Xi == -1:
		new_Ob = Xi
		if Xi == -1:
			new_B = B
		else:
			new_B = B - 1
		local_reward = 0.0
	else:
		new_Ob = Ob
		new_B = B
		local_reward = exp_local_rewards[(Xi, new_Ob)]
	if new_B < 0:
		return None, -inf
	if (Xi, new_Ob, new_B) in mdp_states:
		return mdp_states[(Xi, new_Ob, new_B)][0], mdp_states[(Xi, new_Ob, new_B)][1]
	if Xi < n:
		new_Xi = Xi + 1
		s.select_state, select_reward = e_subset_MDP(True, new_Xi, new_Ob, n, new_B)
		s.skip_state, skip_reward = e_subset_MDP(False, new_Xi, new_Ob, n, new_B)
		s.reward = local_reward + max(select_reward, skip_reward)
		if select_reward > skip_reward:
			s.choice = 1
		else:
			s.choice = 0
	else:
		s.reward = local_reward
	mdp_states[(Xi, new_Ob, new_B)] = (s, s.reward)
	return s, s.reward

#########################################################################################
# IMPLEMENTATION OF VOIDP.
#########################################################################################	
def VoIDP(n, B):
	L = dict()
	for a in range(-1, n + 1):
		for b in range(a + 1, n + 1):
			L[(a, b, 0)] = 0.0
			for j in range(a + 1, b):
				L[(a, b, 0)] += exp_local_rewards[(j, a)]
	for k in range(1, B + 1):
		for a in range(-1, n + 1):
			for b in range(a + 1, n + 1):
				sel = [-inf for j in range(n)]
				for j in range(a + 1, b):
					sel[j] = L[(a, j, 0)] + L[(j, b, k - 1)]
				if b - a > 1:
					L[(a, b, k)] = max(sel)
				else:
					L[(a, b, k)] = L[(a, b, 0)]
	return L[(-1, n, B)]

#########################################################################################
# IMPLEMENTATION OF THE GREEDY HEURISTIC.
# ONE VARIABLE IS ADDED TO THE EVENTUAL SUBSET AT A TIME.
#########################################################################################		
def greedy_heuristic(n, B, baseline):
	observed = [0 for i in range(n)]
	greedy = [baseline]
	new_baseline = baseline
	for i in range(1, B + 1):
		max = new_baseline
		index = -1
		prev_index = -1
		for j in range(n):
			temp = new_baseline
			if observed[j] != 1:
				k = j - 1
				while k > -1 and observed[k] != 1:
					k -= 1
				temp -= exp_local_rewards[(j, k)]
				l = j + 1
				while l < n and observed[l] != 1:
					temp -= (exp_local_rewards[(l, k)] - exp_local_rewards[(l, j)])
					l += 1
				if temp > max:
					max = temp
					index = j
					prev_index = k
		observed[index] = 1
		new_baseline = max
		greedy.append(new_baseline)
	return greedy[len(greedy) - 1]

#########################################################################################
# UNIFORM SPACING HEURISTIC.
#########################################################################################		
def uniform_spacing_heuristic(n, B, baseline):
	uniform = [baseline]
	for i in range(1, B + 1):
		gaps = i + 1
		interval = float(n) / float(gaps)
		index = interval
		budget = i
		observed = [0 for j in range(n)]
		while index < n:
			if budget > 0:
				observed[int(index)] = 1
				budget -= 1
			index += interval
		reward1 = 0.0
		prev_index = -1
		for j in range(n):
			if observed[j] == 0:
				reward1 += exp_local_rewards[(j, prev_index)]
			else:
				prev_index = j
		index = 0.0
		budget = i
		observed = [0 for j in range(n)]
		while index < n:
			if budget > 0:
				observed[int(index)] = 1
				budget -= 1
			index += interval
		reward2 = 0.0
		prev_index = -1
		for j in range(n):
			if observed[j] == 0:
				reward2 += exp_local_rewards[(j, prev_index)]
			else:
				prev_index = j
		uniform.append(max(reward1, reward2))
	uniform[len(uniform) - 1] = 0.0
	return uniform
				

if __name__ == '__main__':
	n = 24
	#########################################################################################
	# PROBABILISTIC INFERENCES.
	#########################################################################################	
	prob_infers = dict()
	start = clock()
	for i in range(-1, n - 1):
		for v1 in get_domain(i):
			for j in range(i + 1, n):
				if i == -1:
					if j - i == 1:
						for v2 in get_domain(j):
							prob_infers[(j, v2, i, v1)] = prior(i, v2)
					else:
						for v2 in get_domain(j):
							temp = 0.0
							for v3 in get_domain(j - 1):
								temp += prob_infers[(j - 1, v3, i, v1)] * cond_prob(j, v2, j - 1, v3)
							prob_infers[(j, v2, i, v1)] = temp
				else:
					if j - i == 1:
						for v2 in get_domain(j):
							prob_infers[(j, v2, i, v1)] = cond_prob(j, v2, i, v1)
					else:
						for v2 in get_domain(j):
							temp = 0.0
							for v3 in get_domain(j - 1):
								temp += prob_infers[(j - 1, v3, i, v1)] * cond_prob(j, v2, j - 1, v3)
							prob_infers[(j, v2, i, v1)] = temp
	#########################################################################################
	# PRE-COMOUTATION OF LOCAL REWARDS.
	#########################################################################################
	for i in range(-1, n - 1):
		for j in range(i + 1, n):
			temp = 0.0
			for v1 in get_domain(i):
				for v2 in get_domain(j):
					if i == -1:
						temp += (prob_infers[(j, v2, i, v1)] * log(prob_infers[(j, v2, i, v1)], 2))
					else:
						temp += (prob_infers[(i, v1, -1, -1)] * prob_infers[(j, v2, i, v1)] * log(prob_infers[(j, v2, i, v1)], 2))
			exp_local_rewards[(j, i)] = temp
	stop = clock()
	pre = stop - start #TIME TO PRECOMPUTE LOCAL REWARDS.
	#########################################################################################
	# CALLS TO eSubsetMDP.
	#########################################################################################
	eSubset = []
	eSubset_time = []
	for B in range(n + 1):
		start = clock()
		s, r = e_subset_MDP(False, -1, -1, n - 1, B)
		stop = clock()
		eSubset.append(r)
		eSubset_time.append(stop - start)
	#########################################################################################
	# CALLS TO VOIDP.
	#########################################################################################
	DP = []
	DP_time = []
	for B in range(n + 1):
		start = clock()
		r = VoIDP(n, B)
		stop = clock()
		DP.append(r)
		DP_time.append(stop - start)
	#########################################################################################
	# CALLS TO THE GREEDY HEURISTIC.
	#########################################################################################
	greedy = []
	greedy_time = []
	for B in range(n + 1):
		start = clock()
		r = greedy_heuristic(n, B, eSubset[0])
		stop = clock()
		greedy.append(r)
		greedy_time.append(stop - start)
	#########################################################################################
	# CALLS TO THE UNIFORM SPACING HEURISTIC.
	#########################################################################################
	uniform = uniform_spacing_heuristic(n, n, eSubset[0])
	#########################################################################################
	# MEASURE THE PERFORMANCE IMPROVEMENT WITH 0 OBSERVATIONS AS BASELINE.
	#########################################################################################
	for i in range(1, n + 1):
		eSubset[i] -= eSubset[0]
		DP[i] -= DP[0]
		greedy[i] -= greedy[0]
		uniform[i] -= uniform[0]
	#########################################################################################
	# OUTPUTS.
	#########################################################################################
	fout = open('chain.txt', 'w')
	for i in range(1, n + 1):
		s = str(i) + '\t' + str(eSubset[i]) + '\t' + str(DP[i]) + '\t' + str(greedy[i]) + '\t' + str(uniform[i]) + '\n'
		fout.write(s)
	fout.close()
	fout = open('chain_time.txt', 'w')
	for i in range(1, n + 1):
		s = str(i) + '\t' + str(eSubset_time[i] + pre) + '\t' + str(DP_time[i] + pre) + '\t' + str(greedy_time[i] + pre) + '\n'
		fout.write(s)
	fout.close()
	