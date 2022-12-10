import numpy as np
# matplotlib.use('TkAgg')
import torch
import math
from load_training_data import get_state_feature, get_action_feature
from rl_algos import value_iteration, build_pi, learn_successor_feature_iter

def is_in_blocked_area(x,y,board):
    '''
    Returns true if the inputted coordinated, (x,y), are in an inaccessible region of the boar

    Input:
    - x,y: input coordinates
    - board: the board configuration
    '''
    val = board[x][y]
    if val == 2 or val == 8:
        return True
    else:
        return False

def find_reward_features(seg,env,use_extended_SF=False,GAMMA=1,seg_length=3):
    '''
    Given a segment, returns the some of reward features for that segment

    Input:
    - seg: the inputted segment, represented as the segments start state and the sequence of actions that follows
    - env: the environment object
    - use_extended_SF: if true, then there is one reward feature for each possible state action pair and each component of the reward function. If false, then there is one reward feature per component of the reward function only.
    - GAMMA: the discount factor
    - seg_length: the segment length, measured as number of transitions

    Output:
    - phi_dis: the discounted sum of reward features for the inputted segment, where the discount factor is GAMMA
    - phi: the undiscounted sum of reward features for the inputted segment
    '''
    seg_ts_x = seg[0][0]
    seg_ts_y = seg[0][1]

    partial_return = 0
    prev_x = seg_ts_x
    prev_y = seg_ts_y
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    if use_extended_SF:
        phi = np.zeros(6+(4 * env.width * env.height))
        phi_dis = np.zeros(6+(4 * env.width * env.height))
    else:
        phi = np.zeros(6)
        phi_dis = np.zeros(6)

    for i in range (1,seg_length+1):
        if env.board[seg_ts_x, seg_ts_y] == 1 or env.board[seg_ts_x, seg_ts_y] == 3 or env.board[seg_ts_x, seg_ts_y] == 7 or env.board[seg_ts_x, seg_ts_y] == 9:
            continue
        if seg_ts_x + seg[i][0] >= 0 and seg_ts_x + seg[i][0] < len(env.board) and seg_ts_y + seg[i][1] >=0 and seg_ts_y + seg[i][1] < len(env.board[0]) and not is_in_blocked_area(seg_ts_x + seg[i][0], seg_ts_y + seg[i][1], env.board):
            seg_ts_x += seg[i][0]
            seg_ts_y += seg[i][1]
        if (seg_ts_x,seg_ts_y) != (prev_x,prev_y):
            dis_state_sf = (GAMMA**(i-1))*get_state_feature(seg_ts_x,seg_ts_y,env)
            state_sf = get_state_feature(seg_ts_x,seg_ts_y,env)
        else:
            #check if we are at terminal state
            if env.board[seg_ts_x, seg_ts_y] == 1 or env.board[seg_ts_x, seg_ts_y] == 3 or env.board[seg_ts_x, seg_ts_y] == 7 or env.board[seg_ts_x, seg_ts_y] == 9:
                dis_state_sf = [0,0,0,0,0,0]
                state_sf = [0,0,0,0,0,0]
            else:
                dis_state_sf = (GAMMA**(i-1))*(get_state_feature(seg_ts_x,seg_ts_y,env)*[1,0,0,0,0,1])
                state_sf = (get_state_feature(seg_ts_x,seg_ts_y,env)*[1,0,0,0,0,1])

        if use_extended_SF:

            #find action index
            for a_i, action_ in enumerate(actions):
                if action_[0] == seg[i][0] and action_[1] == seg[i][1]:
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


        prev_x = seg_ts_x
        prev_y = seg_ts_y
    # print ("--done--\n")
    return phi_dis,phi

def augment_data(X,Y,ytype="scalar"):
    '''
    Augments the preference dataset by flipping the segment pairs and the corresponding preferences

    Input:
    - X: a list of segment pair reward features
    - Y: a list of preferences for each segment pair
    - ytype: if ytype == "scalar", then preferences are scalar values (0 means the first segment is preffered, 1 means the second preference is preffered, 0.5 means they are equally preferred)
             otherwise, the preferences are represented as arrays ([1,0] means the first segment is preffered, [0,1] means the second preference is preffered, [0.5,0.5] means they are equally preferred)

    Output:
    - aX: the augmented list of segment pair reward features
    - aY: the augmented list of preferences
    '''


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
    '''
    If the inputted list of preferences are scalar values, converts them to arrays. Otherwise does nothing. 
    
    Input:
    - Y: a list of preferences for each segment pair
    - ytype: if ytype == "scalar", then preferences are scalar values (0 means the first segment is preffered, 1 means the second preference is preffered, 0.5 means they are equally preferred)
             otherwise, the preferences are represented as arrays ([1,0] means the first segment is preffered, [0,1] means the second preference is preffered, [0.5,0.5] means they are equally preferred)
    Output:
    - formatted_y: a tensor containing preferences represented as arrays

    '''
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
    '''
    Converts X, a list of segment pair reward features, to a float tensor
    '''
    return torch.tensor(X,dtype=torch.float)

def format_X_pr (X):
    '''
    Returns a list of difference in partial return values for segment pairs

    Input:
    - X is a list of segment pair stastics where X[i] = [difference in partial return, difference in start state value, difference in end state value]
    Output
    - formatted_X: a list of difference in partial returns for segment pairs
    '''
    formatted_X= []
    for x in X:
        formatted_X.append([x[0]])
    return torch.tensor(formatted_X,dtype=torch.float)

def format_X_regret (X):
    '''
    Returns a list of difference in regret values for segment pairs

    Input:
    - X is a list of segment pair stastics where X[i] = [difference in partial return, difference in start state value, difference in end state value]
    Output
    - formatted_X: a list of difference in regret values for segment pairs
    '''

    formatted_X= []
    for x in X:
        formatted_X.append([x[0]+ x[1] - x[2]])# + x[1] - x[2]
    return torch.tensor(formatted_X,dtype=torch.float)

def format_X_full (X):
    '''
    Converts X to a float tensor
    '''
    return torch.tensor(X,dtype=torch.float)

def sigmoid(val):
    '''
    Sigmoid function
    '''
    return 1 / (1 + math.exp(-val))

def get_pref(arr,include_eps = True):
    '''
    Given a segment pair stastic (partial return, regret, etc.), returns the error-free preference for that segment pair
    '''
    if (arr[0] > arr[1]):
        res = [1,0]
    elif (arr[1] > arr[0]):
        res = [0,1]
    else:
        res = [0.5,0.5]
    return res

def disp_mmv(arr,title,axis):
    '''
    Prints the mean, median, and variance for an array
    '''
    print ("Mean " + title + ": " + str(np.mean(arr,axis=axis)))
    print ("Median " + title + ": " + str(np.median(arr,axis=axis)))
    print (title + " Variance: " + str(np.var(arr,axis=axis)))

def remove_gt_succ_feat(succ_feats, succ_feats_q, gt_rew_vec):
    '''
    Removes the successor feature of the optimal policy under the ground truth reward function from a list of successor features

    Input:
    - succ_feats: a list of state successor features
    - succ_feats_q: a list of state, action successor features
    - gt_rew_vec: the ground truth reward function

    Ouput:
    - mod_succ_feats: a list of state successor features without the successor feature of the optimal policy under the ground truth reward function
    - mod_succ_feats_q: a list of state action successor features without the successor feature of the optimal policy under the ground truth reward function
    '''
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



def remove_gt_succ_feat(succ_feats, succ_feats_q, action_succ_feats, gt_rew_vec,GAMMA):
    '''
    Removes the successor feature of the optimal policy under the ground truth reward function from a list of successor features

    Input:
    - succ_feats: a list of state successor features
    - succ_feats_q: a list of state, action successor features
    - action_succ_feats: a list of action successor features
    - gt_rew_vec: the ground truth reward function
    - GAMMA: the discount value

    Ouput:
    - mod_succ_feats: a list of state successor features without the successor feature of the optimal policy under the ground truth reward function
    - mod_succ_feats_q: a list of state action successor features without the successor feature of the optimal policy under the ground truth reward function
    - mod_action_succ_feats: a list of action successor features without the successor feature of the optimal policy under the ground truth reward function
    '''
    vec = np.array(gt_rew_vec)
    V,Qs = value_iteration(rew_vec = vec,GAMMA=GAMMA)
    pi = build_pi(Qs)
    gt_succ_feat, gt_action_succ_feat, gt_q_succ_feat = learn_successor_feature_iter(pi,GAMMA,rew_vec = vec)
    mod_succ_feats = []
    mod_succ_feats_q = []
    mod_action_succ_feats = []

    for succ_feat, succ_feat_q , action_succ_feat in zip(succ_feats, succ_feats_q, action_succ_feats):
        if not np.array_equal(gt_succ_feat, succ_feat):
            mod_succ_feats.append(succ_feat)
        if not np.array_equal(gt_q_succ_feat, succ_feat_q):
            mod_succ_feats_q.append(succ_feat_q)
        if not np.array_equal(gt_action_succ_feat, action_succ_feat):
            mod_action_succ_feats.append(action_succ_feat)

    return mod_succ_feats, mod_succ_feats_q, mod_action_succ_feats