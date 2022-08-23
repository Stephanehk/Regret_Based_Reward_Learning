import numpy as np
# matplotlib.use('TkAgg')
import torch
import math
from load_training_data import get_state_feature, get_action_feature
from rl_algos import value_iteration, build_pi, learn_successor_feature_iter

def find_reward_features(traj,env,use_extended_SF=False,GAMMA=1,traj_length=3):
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    if use_extended_SF:
        phi = np.zeros(6+(4 * env.width * env.height))
        phi_dis = np.zeros(6+(4 * env.width * env.height))
    else:
        phi = np.zeros(6)
        phi_dis = np.zeros(6)

    for i in range (1,traj_length+1):
        if env.board[traj_ts_x, traj_ts_y] == 1 or env.board[traj_ts_x, traj_ts_y] == 3 or env.board[traj_ts_x, traj_ts_y] == 7 or env.board[traj_ts_x, traj_ts_y] == 9:
            continue
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(env.board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(env.board[0]) and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], env.board):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]
        if (traj_ts_x,traj_ts_y) != (prev_x,prev_y):
            dis_state_sf = (GAMMA**(i-1))*get_state_feature(traj_ts_x,traj_ts_y,env)
            state_sf = get_state_feature(traj_ts_x,traj_ts_y,env)
        else:
            #check if we are at terminal state
            if env.board[traj_ts_x, traj_ts_y] == 1 or env.board[traj_ts_x, traj_ts_y] == 3 or env.board[traj_ts_x, traj_ts_y] == 7 or env.board[traj_ts_x, traj_ts_y] == 9:
                dis_state_sf = [0,0,0,0,0,0]
                state_sf = [0,0,0,0,0,0]
            else:
                dis_state_sf = (GAMMA**(i-1))*(get_state_feature(traj_ts_x,traj_ts_y,env)*[1,0,0,0,0,1])
                state_sf = (get_state_feature(traj_ts_x,traj_ts_y,env)*[1,0,0,0,0,1])

        if use_extended_SF:

            #find action index
            for a_i, action_ in enumerate(actions):
                if action_[0] == traj[i][0] and action_[1] == traj[i][1]:
                    action_index = a_i


            if env.board[prev_x, prev_y] == 1 or env.board[prev_x, prev_y] == 3 or env.board[prev_x, prev_y] == 7 or env.board[prev_x, prev_y] == 9:
                dis_action_sf = np.zeros(env.height*env.width*4)
            else:
                dis_action_sf =(GAMMA**(i-1))*get_action_feature(prev_x, prev_y, action_index,env=env)
            dis_state_sf = list(dis_state_sf)
            dis_state_sf.extend(dis_action_sf)

            if env.board[prev_x, prev_y] == 1 or env.board[prev_x, prev_y] == 3 or env.board[prev_x, prev_y] == 7 or env.board[prev_x, prev_y] == 9:
                action_sf = np.zeros(env.height*env.width*4)
            else:
                action_sf =get_action_feature(prev_x, prev_y, action_index,env=env)

            # print ("action_sf: " + str(action_sf))
            state_sf = list(state_sf)
            state_sf.extend(action_sf)

        phi_dis += dis_state_sf
        phi+= state_sf


        prev_x = traj_ts_x
        prev_y = traj_ts_y
    # print ("--done--\n")
    return phi_dis,phi

def augment_data(X,Y,ytype="scalar"):
    aX = []
    ay = []
    for x,y in zip(X,Y):
        aX.append(x)
        ay.append(y)

        neg_x = [x[1],x[0]]
        aX.append(neg_x)
        if ytype == "scalar":
            ay.append(1-y)
        else:
            ay.append([y[1],y[0]])
    # return np.array(aX), np.array(ay)
    return aX, ay

def format_y(Y,ytype="scalar"):
    formatted_y = []
    if ytype=="scalar":
        for y in Y:
            if y == 0:
                formatted_y.append(np.array([1,0]))
            elif y == 1:
                formatted_y.append(np.array([0,1]))
            elif y == 0.5:
                formatted_y.append(np.array([0.5,0.5]))
    else:
        formatted_y = Y
    return torch.tensor(formatted_y,dtype=torch.float)


def format_X(X):
    return torch.tensor(X,dtype=torch.float)

def format_X_pr (X):
    formatted_X= []
    for x in X:
        formatted_X.append([x[0]])
    return torch.tensor(formatted_X,dtype=torch.float)

def format_X_regret (X):
    formatted_X= []
    for x in X:
        formatted_X.append([x[0]+ x[1] - x[2]])# + x[1] - x[2]
    return torch.tensor(formatted_X,dtype=torch.float)

def format_X_full (X):
    return torch.tensor(X,dtype=torch.float)

def sigmoid(val):
    return 1 / (1 + math.exp(-val))

def get_pref(arr,include_eps = True):
    if (arr[0] > arr[1]):
        res = [1,0]
    elif (arr[1] > arr[0]):
        res = [0,1]
    else:
        res = [0.5,0.5]
    return res

def disp_mmv(arr,title,axis):
    print ("Mean " + title + ": " + str(np.mean(arr,axis=axis)))
    print ("Median " + title + ": " + str(np.median(arr,axis=axis)))
    print (title + " Variance: " + str(np.var(arr,axis=axis)))

def remove_gt_succ_feat(succ_feats, succ_feats_q, gt_rew_vec):
    vec = np.array(gt_rew_vec)
    V,Qs = value_iteration(rew_vec = vec,GAMMA=0.999)
    pi = build_pi(Qs)
    gt_succ_feat, gt_action_succ_feats, gt_q_succ_feat = learn_successor_feature_iter(pi,0.999,rew_vec = vec)
    mod_succ_feats = []
    mod_succ_feats_q = []

    for succ_feat, succ_feat_q in zip(succ_feats, succ_feats_q):
        if not np.array_equal(gt_succ_feat, succ_feat):
            mod_succ_feats.append(succ_feat)
        if not np.array_equal(gt_q_succ_feat, succ_feat_q):
            mod_succ_feats_q.append(succ_feat_q)

    return mod_succ_feats, mod_succ_feats_q

