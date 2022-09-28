import numpy as np
from collections import defaultdict
# import plotting
from grid_world import GridWorldEnv
import random
import matplotlib.pyplot as plt

import random_policy_data
from rl_algos import learn_successor_feature_iter, value_iteration,build_pi

changed_gt_rew_vec = False

vec = np.array([-1,50,-50,1,-1,-2])
V,Qs = value_iteration(rew_vec = vec,GAMMA=0.999)
pi = build_pi(Qs)
gt_succ_feat,_, gt_q_succ_feat = learn_successor_feature_iter(pi,0.999,rew_vec = vec)


def get_random_reward_vector(gt_rew_vec):
    '''
    Compute random reward vector
    '''
    if gt_rew_vec is None:
        space = [-1,50,-50,1,-1,-2]
    else:
        space = [50,-50,1,-1,-2,0,-10,10,5]
    vector = []
    for i in range(6):
        s = random.choice(space)
        # space.remove(s)
        vector.append(s)
    return np.array(vector)

def generate_random_policy(GAMMA,env=None,gt_rew_vec=None):
    '''
    Generate a policy for a random reward vector
    '''
    vec = get_random_reward_vector(gt_rew_vec)
    # vec = np.array([-1,50,-50,1,-1,-2])
    V,Qs = value_iteration(rew_vec = vec,GAMMA=GAMMA,env=env)
    # follow_policy(Qs, 1000,viz_policy=True)
    pi = build_pi(Qs,env=env)
    # psi_og = learn_successor_feature(Qs,V,GAMMA,rew_vec = vec)
    succ_feat,sa_succ_feat,gt_q_succ_feat = learn_successor_feature_iter(pi,GAMMA,rew_vec = vec,env=env)

    return succ_feat, sa_succ_feat,pi, gt_q_succ_feat

def is_arr_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def generate_all_policies(n_policies,GAMMA,env=None,gt_rew_vec=None):
    '''
    Generate a list of policies from random reward vectors
    '''
    succ_feats = []
    gt_q_succ_feats = []
    sa_succ_feats = []
    pis = []
    i = 0
    n_duplicates = 0 #makes sure we do not try and generate more unique policies than exist
    while i < n_policies and n_duplicates < 100:
        i+=1
        succ_feat, sa_succ_feat, pi, gt_q_succ_feat = generate_random_policy(GAMMA,env,gt_rew_vec)
        if is_arr_in_list(succ_feat, succ_feats):
            i-=1
            n_duplicates+= 1
        else:
            # print ("generated policy: " + str(len(pis)))
            succ_feats.append(succ_feat)
            gt_q_succ_feats.append(gt_q_succ_feat)
            sa_succ_feats.append(sa_succ_feat)
            pis.append(pi)
    return succ_feats,sa_succ_feats, pis, gt_q_succ_feats


def calc_advantage(states,actions,gt_rew_vec=None,env=None):
    '''
    Calculate advantage under V* and Q*
    '''
    id_ =1
    if gt_rew_vec is None or env is None:
        advantage = 0
        for state,action in zip(states,actions):
            x,y = state
            advantage += random_policy_data.V[x][y] - random_policy_data.Qs[x][y][action]
        return -advantage
    else:
        if not random_policy_data.changed_gt_rew_vec:
            V,Qs = value_iteration(rew_vec = np.array(gt_rew_vec),GAMMA=0.999,env=env)
            pi = build_pi(Qs,env=env)
            gt_succ_feat,_,gt_q_succ_feat = learn_successor_feature_iter(pi,0.999,rew_vec = gt_rew_vec,env=env)
            np.save("gt_succ_feat_"+str(id)+".npy",gt_succ_feat)
            np.save("V_"+str(id_)+".npy",V)
            np.save("Qs_"+str(id_)+".npy",Qs)
            # global changed_gt_rew_vec
            random_policy_data.changed_gt_rew_vec = True
            print ("CHANGED GT_REW_VEC")
        else:
            V = np.load("V_"+str(id_)+".npy")
            Qs = np.load("Qs_"+str(id_)+".npy")

        advantage = 0
        for state,action in zip(states,actions):
            x,y = state
            advantage += V[x][y] - Qs[x][y][action]
        return -advantage

def calc_value(state,gt_rew_vec=None,env=None):
    '''
    Calculate value under V*
    '''
    id_ = 1
    if gt_rew_vec is None or env is None:
        w = [-1,50,-50,1,-1,-2]
        x,y = state
        return np.dot(random_policy_data.gt_succ_feat[x][y],w)
    else:
        w = gt_rew_vec
        x,y = state
        if not random_policy_data.changed_gt_rew_vec:
            V,Qs = value_iteration(rew_vec = np.array(gt_rew_vec),GAMMA=0.999,env=env)
            pi = build_pi(Qs,env=env)
            gt_succ_feat,_,gt_q_succ_feat = learn_successor_feature_iter(pi,0.999,rew_vec = gt_rew_vec,env=env)
            np.save("gt_succ_feat_"+str(id_)+".npy",gt_succ_feat)
            np.save("V_"+str(id_)+".npy",V)
            np.save("Qs_"+str(id_)+".npy",Qs)
            # global changed_gt_rew_vec
            random_policy_data.changed_gt_rew_vec = True
            return np.dot(gt_succ_feat[x][y],w)
        else:
            gt_succ_feat = np.load("gt_succ_feat_"+str(id_)+".npy",allow_pickle=True)
            return np.dot(gt_succ_feat[x][y],w)

