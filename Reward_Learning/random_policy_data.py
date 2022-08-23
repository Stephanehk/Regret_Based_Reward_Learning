from rl_algos import build_random_policy, learn_successor_feature_iter, iterative_policy_evaluation,value_iteration,build_pi
import numpy as np

changed_gt_rew_vec = False
vec = np.array([-1,50,-50,1,-1,-2])
V,Qs = value_iteration(rew_vec = vec,GAMMA=0.999)
pi = build_pi(Qs)
gt_succ_feat, gt_action_succ_feats, gt_q_succ_feat = learn_successor_feature_iter(pi,0.999,rew_vec = vec)

random_pi = build_random_policy()
V_under_random_pi = iterative_policy_evaluation(random_pi,rew_vec = np.array(vec))
random_avg_return = np.sum(V_under_random_pi)/92

print ("random policies avg return: ")

print (random_avg_return)
