import numpy as np
import torch.nn as nn
import pickle
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import random
from datetime import datetime
import os
import argparse

from load_training_data import get_all_statistics,get_all_statistics_aug_human
from rl_algos import value_iteration, get_gt_avg_return,build_pi,iterative_policy_evaluation,learn_successor_feature_iter,build_random_policy
from generate_random_policies import calc_value, calc_advantage
import random_policy_data
from utils import augment_data, find_reward_features, format_X, format_y, sigmoid, get_pref, disp_mmv



parser = argparse.ArgumentParser(description='PyTorch RL trainer')

parser.add_argument('--keep_ties', action='store_true', default=True,
                    help='Keep indifferent preferences in preference dataset')
parser.add_argument('--include_dif_seg_lengths', action='store_true', default=True,
                    help='Include segment pairs where one segment terminates earlier than the other')
parser.add_argument('--n_prob_samples', default=1, type=int,
                    help='Number of times a preference label is sampled from each segment pair when using a stochastic preference model')
parser.add_argument('--n_prob_iters', default=30, type=int,
                    help='Number of trials for learning a reward function when using a stochastic preference model')
parser.add_argument('--GAMMA', default=0.999, type=float,
                    help='Discount factor for preference labeling and reward learning/evaluation')
parser.add_argument('--mode', default="deterministic", type=str,
                    help='Either deterministic (for error-free synthetic preference dataset), sigmoid (for stochastic synthetic preference dataset), or deterministic_user_data (for the human preference dataset)')
parser.add_argument('--LR', default=0.5, type=float,
                    help='Learning rate')
parser.add_argument('--N_ITERS', default=5000, type=int,
                    help='Number of training iterations')
parser.add_argument('--use_random_MDPs',  action='store_true', default=False,
                    help='Run expirements on set of 100 random MDPs instead of on the original delivery domain')
parser.add_argument('--use_random_MDPs_n_length_segs',  action='store_true', default=False,
                    help='Run expirements on set of 100 random MDPs instead of on the original delivery domain')
parser.add_argument('--partition_human_data',  action='store_true', default=False,
                    help='Run expirements on partitions of the human preference dataset instead of on the entire dataset')
parser.add_argument('--preference_model',  type=str, default="regret",
                    help='preference model for how we generate synthetic preferences (pr for partial return model, er for regret model)')
parser.add_argument('--preference_assum',  type=str, default="regret",
                    help='preference model for how we learn a reward function from preferences (pr for partial return model, er for regret model)')
args = parser.parse_args()

keep_ties = args.keep_ties
n_prob_samples = args.n_prob_samples
n_prob_iters = args.n_prob_iters

GAMMA=args.GAMMA
include_dif_seg_lengths = args.include_dif_seg_lengths

mode = args.mode
LR = args.LR
N_ITERS = args.N_ITERS
optimizer_add = "none"

use_random_MDPs = args.use_random_MDPs
use_random_MDPs_n_length_segs = args.use_random_MDPs_n_length_segs

use_extended_SF = False
run_temp_exp = False
include_actions = False
partition_human_data = args.partition_human_data
preference_model = args.preference_model #how we generate prefs
preference_assum = args.preference_assum #how we learn prefs

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


if preference_assum == "regret" and not use_random_MDPs:
    succ_feats = np.load("succ_feats_no_gt.npy",allow_pickle=True)
    pis = np.load("pis_no_gt.npy",allow_pickle=True)
    succ_q_feats = None

def clean_y(X,R,Y,sess):
    '''
    Input:
        - X: a list of each segment pairs features
        - R: a list of each segment pairs partial return
        - Y: a list of human preferences for each segment pairs
        - sess: a list of the the start and end states for each segment pair

    Output:
        - the properly formatted segment features and preferences 
    '''

    formatted_y = []
    out_X = []

    synth_formatted_y = []
    synth_y_dist = []
    synth_out_X = []

    for x,r,y,ses in zip(X,R,Y,sess):
        x = [list(x[0]),list(x[1])]
        #change x to include start end state for each segment
        if preference_assum == "regret":
            x[0] = list(x[0])
            x[0].extend([ses[0][0][0],ses[0][0][1], ses[0][1][0],ses[0][1][1]])
            x[1] = list(x[1])
            x[1].extend([ses[1][0][0],ses[1][0][1], ses[1][1][0],ses[1][1][1]])
            x = [x[0],x[1]]
        if y == 0:
            formatted_y.append(np.array([1,0]))
            out_X.append(x)
        elif y == 1:
            formatted_y.append(np.array([0,1]))
            out_X.append(x)
        elif y == 0.5:
            formatted_y.append(np.array([0.5,0.5]))
            out_X.append(x)
        elif y == None:
           assert False
        else:
            formatted_y.append(np.array(y))
            out_X.append(x)

    loss_coef = 1
    return out_X,formatted_y,synth_out_X,synth_formatted_y,synth_y_dist, loss_coef


def get_gt_regret (x,r,t1_ss=None,t1_es=None,t2_ss=None,t2_es=None,actions=None,states=None,gt_rew_vec=None,env=None):
    '''
    Calculates the ground truth regret for a single segment pairs

    Input:
        - x: a list of each segment pairs features
        - r: a list of each segment pairs partial return
        - t1_ss, t1_es, ...: the start and end state for each segment
        - actions/states: a list of state action pairs for each segment, only used for stochastic MDPs
        - gt_rew_vec: the ground truth reward vector, None if using the default delivery domain 
        - env: the environment, None if using the default delivery domain 

    Output:
        - ground truth regret for each segment
    '''
    #calculates gt expected return
    if gt_rew_vec is not None:
        w = gt_rew_vec
    else:
        w = [-1,50,-50,1,-1,-2]
    if (torch.is_tensor(x)):
        x = x.detach().cpu().numpy()


    if t1_ss == None:
        assert False
        t1_ss = [int(x[0][6]), int(x[0][7])]
        t1_es = [int(x[0][8]), int(x[0][9])]

        t2_ss = [int(x[1][6]), int(x[1][7])]
        t2_es = [int(x[1][8]), int(x[1][9])]

    if actions is None and include_actions:
        assert False
    elif actions is not None and include_actions:
        seg1_actions = actions[0]
        seg2_actions = actions[1]

        seg1_states = states[0]
        seg2_states = states[1]

        assert tuple(seg1_states[0]) == tuple(t1_ss)
        assert tuple(seg1_states[-1]) == tuple(t1_es)
        assert tuple(seg2_states[0]) == tuple(t2_ss)
        assert tuple(seg2_states[-1]) == tuple(t2_es)


    if include_actions:
        
        r1_cer = calc_advantage(seg1_states,seg1_actions,gt_rew_vec,env)
        r2_cer = calc_advantage(seg2_states,seg2_actions,gt_rew_vec,env)

    else:
        x = np.array(x)
        # r = np.dot(x[:,0:6],w)

        r1_cer = r[0] + calc_value(t1_es,gt_rew_vec,env) - calc_value(t1_ss,gt_rew_vec,env)
        r2_cer = r[1] + calc_value(t2_es,gt_rew_vec,env) - calc_value(t2_ss,gt_rew_vec,env)
    r1_cer = np.round(r1_cer,2)
    r2_cer = np.round(r2_cer,2)
    return r1_cer, r2_cer


def generate_synthetic_prefs(pr_X,rewards,sess,actions,states,mode,gt_rew_vec=[-1,50,-50,1,-1,-2],env=None):
    '''
    Generates synthetic preferences using the model specified by preference_assum

    Input:
        - pr_X: a list of each segment pairs features
        - rewards: a list of each segment pairs partial return
        - sess: a list of the the start and end states for each segment pair
        - actions/states: a list of state action pairs for each segment, only used for stochastic MDPs
        - mode: sigmoid (for stochastic preferences) or deterministic (for error-free preferences)
        - gt_rew_vec: the ground truth reward vector, None if using the default delivery domain 
        - env: the environment, None if using the default delivery domain 

    Output:
        - a list of syntheticallt generated preferences and their corresponding preference pair features
    '''
    synth_y = []
    non_redundent_pr_X = []
    expected_returns = []
    n_removed = 0
   
    index = 0

    #quick error handling
    if actions is None:
        actions = [None for i in range(len(pr_X))]
        states = [None for i in range(len(pr_X))]

    assert len(sess) == len(pr_X) == len(rewards)

    for r,x,ses in zip(rewards,pr_X,sess):
        x = list(x)
        if preference_model == "pr" and preference_assum == "pr":
            x_f = [list(x[0][0:6]),list(x[1][0:6])]
            x_orig = [list(x[0]), list(x[1])]
        #change x to include start end state for each trajectory
        if preference_model == "regret" and preference_assum == "regret":
            if include_actions:
                #actions[index],states[index]
                x[0] = list(x[0])
                x[0].extend(actions[index][0])
                x[0].extend(np.array(states[index][0][0:len(states[index][0])-1]).flatten())

                x[1] = list(x[1])
                x[1].extend(actions[index][1])
                x[1].extend(np.array(states[index][1][0:len(states[index][1])-1]).flatten())

                x_f = [x[0],x[1]]
                x_orig = [list(x[0]), list(x[1])]

                if len(x_f[0]) != len(x_f[1]):
                    print (actions[index])
                    print (states[index])
                    assert False

            else:
                x[0] = list(x[0])
                x[0].extend([ses[0][0][0],ses[0][0][1], ses[0][1][0],ses[0][1][1]])

                x[1] = list(x[1])
                x[1].extend([ses[1][0][0],ses[1][0][1], ses[1][1][0],ses[1][1][1]])
                x_f = [x[0],x[1]]
                x_orig = [list(x[0]), list(x[1])]
        if preference_model == "pr" and preference_assum == "regret":
            if include_actions:
                assert False
                #actions[index],states[index]
            else:
                x[0] = list(x[0])
                x[0].extend([ses[0][0][0],ses[0][0][1], ses[0][1][0],ses[0][1][1]])

                x[1] = list(x[1])
                x[1].extend([ses[1][0][0],ses[1][0][1], ses[1][1][0],ses[1][1][1]])
                x_f = [x[0],x[1]]
                x_orig = [list(x[0]), list(x[1])]
        if preference_model == "regret" and preference_assum == "pr":
            x_f = [list(x[0]),list(x[1])]
            x_orig = [list(x[0]), list(x[1])]

        t1_ss = [int(ses[0][0][0]), int(ses[0][0][1])]
        t1_es = [int(ses[0][1][0]), int(ses[0][1][1])]

        t2_ss = [int(ses[1][0][0]), int(ses[1][0][1])]
        t2_es = [int(ses[1][1][0]), int(ses[1][1][1])]

        if mode == "sigmoid":
            if preference_model == "regret":
                r1_er, r2_er = get_gt_regret (x,r,t1_ss,t1_es,t2_ss,t2_es,actions[index],states[index],gt_rew_vec,env)

            if preference_model == "pr" and not keep_ties and r[1] == r[0]:
                continue

            if preference_model == "regret" and not keep_ties and r1_er == r2_er:
                continue

            for n_samp in range(n_prob_samples):
                if preference_model == "pr":
                    r1_prob = sigmoid((r[0]-r[1])/1)
                    r2_prob = sigmoid((r[1]-r[0])/1)
                elif preference_model == "regret":
                    r1_prob = sigmoid((r1_er-r2_er)/1)
                    r2_prob = sigmoid((r2_er-r1_er)/1)
                num = np.random.choice([1,0], p=[r1_prob,r2_prob])
                if num == 1:
                    pref = [1,0]
                elif num == 0:
                    pref = [0,1]
                synth_y.append(pref)
                non_redundent_pr_X.append(x_orig)
        else:
            if preference_model == "regret":
                r1_er, r2_er = get_gt_regret (x,r,t1_ss,t1_es,t2_ss,t2_es,actions[index],states[index],gt_rew_vec,env)
                pref = get_pref([r1_er, r2_er])
            else:
                pref = get_pref(r)

            if pref == [0.5,0.5] and not keep_ties:
                continue

            if preference_model == "regret":
                expected_returns.append([r1_er, r2_er])
            synth_y.append(pref)
            non_redundent_pr_X.append(x_orig)
        index+=1
    print ("removed " + str(n_removed) + " duplicate segment pairs")
    return non_redundent_pr_X, synth_y,expected_returns

def reward_pred_loss(output, target):
    '''
    Calculates cross entropy loss between predicted and ground truth preferences
    '''
    batch_size = output.size()[0]
    output = torch.squeeze(output)
    output = torch.log(output)
    res = torch.mul(output,target)
    return -torch.sum(res)


class RewardFunctionPR(torch.nn.Module):
    '''
    The partial return reward learning model
    '''
    def __init__(self,GAMMA, n_features=6):
        super(RewardFunctionPR, self).__init__()
        self.n_features = n_features
        self.GAMMA = GAMMA
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False)


    def forward(self, phi):
        pr = torch.squeeze(self.linear1(phi))
        left_pred = torch.sigmoid(torch.subtract(pr[:,0:1],pr[:,1:2]))
        right_pred = torch.sigmoid(torch.subtract(pr[:,1:2],pr[:,0:1]))
        phi_logit = torch.stack([left_pred,right_pred],axis=1)
        return phi_logit

class RewardFunctionRegret(torch.nn.Module):
    '''
    The regret reward learning model
    '''
    def __init__(self,GAMMA,succ_feats,preference_weights, n_features=6,include_actions=False,succ_q_feats=None):
        super(RewardFunctionRegret, self).__init__()
        self.n_features = n_features
        self.GAMMA = GAMMA
        self.succ_feats = torch.tensor(succ_feats,dtype=torch.double).to(device)

        if succ_q_feats is not None:
            self.succ_q_feats = torch.tensor(succ_q_feats,dtype=torch.double).to(device)

        self.include_actions = include_actions
        # self.succ_feats_gt = torch.tensor(succ_feats_gt,dtype=torch.double)
        # self.w = torch.nn.Parameter(torch.tensor(np.zeros(n_features).T,dtype = torch.float,requires_grad=True))
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False).double()

        self.softmax = torch.nn.Softmax(dim=1)
        
        #optionally set weights for the deterministic regret model
        if preference_weights is not None:
            self.rw = preference_weights[0][0]
            self.v_stw = preference_weights[0][1]
            self.v_s0w = preference_weights[0][2]
        else:
            self.rw = 1
            self.v_stw = 1
            self.v_s0w = 1

        self.T = 0.001
        self.dummy_mdp = True

    def get_qs(self,state,action):
        '''
        Approximate Q* using successor feautres
        '''
        selected_succ_feats =[self.succ_q_feats[:,x,y,a].double() for (x,y),a in zip(state.long(),action.long())]
        selected_succ_feats = torch.stack(selected_succ_feats)

        qs = self.linear1(selected_succ_feats)
        q_pi_approx = torch.sum(torch.mul(self.softmax(qs/self.T),qs),dim = 1)

        q_pi_approx = torch.squeeze(q_pi_approx)
        return q_pi_approx

    def get_vals(self,cords):
        '''
        Approximate V* using successor feautres
        '''
        selected_succ_feats =[self.succ_feats[:,x,y].double() for x,y in cords.long()]
        selected_succ_feats = torch.stack(selected_succ_feats)
        vs = self.linear1(selected_succ_feats)
        v_pi_approx = torch.sum(torch.mul(self.softmax(vs/self.T),vs),dim = 1)

        v_pi_approx = torch.squeeze(v_pi_approx)
        return v_pi_approx

    def forward(self, phi):

        if self.include_actions:
            a1 = torch.squeeze(phi[:,:,6:7])
            a2 = torch.squeeze(phi[:,:,7:8])
            a3 = torch.squeeze(phi[:,:,8:9])

            s1_x = torch.squeeze(phi[:,:,9:10])
            s1_y = torch.squeeze(phi[:,:,10:11])
            s1 = torch.stack([s1_x,s1_y], dim=1)

            s2_x = torch.squeeze(phi[:,:,11:12])
            s2_y = torch.squeeze(phi[:,:,12:13])
            s2 = torch.stack([s2_x,s2_y], dim=1)

            s3_x = torch.squeeze(phi[:,:,13:14])
            s3_y = torch.squeeze(phi[:,:,14:15])
            s3 = torch.stack([s3_x,s3_y], dim=1)

            q1 = self.get_qs(s1,a1)
            v1 = self.get_vals(s1)
            adv1 = torch.subtract(v1,q1)
            left_adv1 = adv1[:,0:1]
            right_adv1 = adv1[:,1:2]

            q2 = self.get_qs(s2,a2)
            v2 = self.get_vals(s2)
            adv2 = torch.subtract(v2,q2)
            left_adv2 = adv2[:,0:1]
            right_adv2 = adv2[:,1:2]

            left_delta_er = torch.add(left_adv1, left_adv2)
            right_delta_er = torch.add(right_adv1, right_adv2)


            q3 = self.get_qs(s3,a3)
            v3 = self.get_vals(s3)
            adv3 = torch.subtract(v3,q3)
            left_adv3 = adv3[:,0:1]
            right_adv3 = adv3[:,1:2]

            left_delta_er = -torch.add(left_delta_er, left_adv3)
            right_delta_er = -torch.add(right_delta_er, right_adv3)
        else:
            pr = torch.squeeze(self.linear1(phi[:,:,0:self.n_features].double()))
            ss_x = torch.squeeze(phi[:,:,self.n_features:self.n_features+1])
            ss_y = torch.squeeze(phi[:,:,self.n_features+1:self.n_features+2])
            ss_cord_pairs = torch.stack([ss_x,ss_y], dim=1)


            es_x = torch.squeeze(phi[:,:,self.n_features+2:self.n_features+3])
            es_y = torch.squeeze(phi[:,:,self.n_features+3:self.n_features+4])
            es_cord_pairs = torch.stack([es_x,es_y], dim=1)


            #build list of succ fears for start/end states
            v_ss = self.get_vals(ss_cord_pairs)
            v_es = self.get_vals(es_cord_pairs)


            left_pr = pr[:,0:1]
            right_pr = pr[:,1:2]

            left_vf_ss = v_ss[:,0:1]
            right_vf_ss = v_ss[:,1:2]

            left_vf_es = v_es[:,0:1]
            right_vf_es = v_es[:,1:2]


            #apply weights learned from logistic regression (if it exists)
            left_pr = torch.multiply(left_pr, self.rw)
            right_pr = torch.multiply(right_pr, self.rw)

            left_vf_ss = torch.multiply(left_vf_ss, self.v_s0w)
            right_vf_ss = torch.multiply(right_vf_ss, self.v_s0w)

            left_vf_es = torch.multiply(left_vf_es, self.v_stw)
            right_vf_es = torch.multiply(right_vf_es, self.v_stw)

            #calculate change in expected return
            left_delta_v = torch.subtract(left_vf_es, left_vf_ss)
            right_delta_v = torch.subtract(right_vf_es, right_vf_ss)

            left_delta_er = torch.add(left_pr, left_delta_v)
            right_delta_er = torch.add(right_pr, right_delta_v)

        left_pred = torch.sigmoid(torch.subtract(left_delta_er, right_delta_er))
        right_pred = torch.sigmoid(torch.subtract(right_delta_er, left_delta_er))

        phi_logit = torch.stack([left_pred,right_pred],axis=1)

        return phi_logit


def run_single_set(model, optimizer, X_train, y_train, loss_coef):
    '''
    Runs a single training iteration for the given preference model on a batch of data 
    '''
    model.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(X_train)
    y_pred = torch.clamp(y_pred,min=1e-35,max=None)#prevents prob pred of 0

    loss = reward_pred_loss(y_pred, y_train)
    batch_size = y_pred.size()[0]

    loss /= (batch_size)
    return loss, optimizer, model

def train(aX, ay, loss_coef = None, plot_loss=True,preference_weights=None,gt_rew_vec=None,env=None):
    '''
    Trains the preference model

    Input:
        - aX: a list of each preference pairs features
        - ay: a list of preferences for each segment pair
        - loss_coef: optional coefficient for loss function (default is 1)
        - plot_loss: plot loss after training
        - preference_weights: optional weights for deterministic regret model
        - gt_rew_vec: the ground truth reward vector, None if using the default delivery domain 
        - env: the environment, None if using the default delivery domain 
    '''
    torch.manual_seed(0) #for exact reproducibility
    X_train = format_X(aX)
    y_train = format_y(ay,"arr")

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    if use_extended_SF:
        if env is not None:
            n_feats = 4*env.width*env.height + 6
        else:
            n_feats = 406
    else:
        n_feats = 6

    if preference_assum == "pr":
        model = RewardFunctionPR(GAMMA,n_features=n_feats)
    elif preference_assum == "regret":
        model = RewardFunctionRegret(GAMMA,succ_feats,preference_weights,include_actions=include_actions,succ_q_feats=succ_q_feats,n_features=n_feats)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []


    loss = None
    best_total_loss = float("inf") #best loss over entire training cycle

    for param in model.parameters():
        best_weights = param.detach().cpu().numpy()[0]

    for epoch in range(N_ITERS):
        loss, optimizer, model = run_single_set(model, optimizer, X_train, y_train, loss_coef)
        if loss.detach().cpu().numpy() < best_total_loss:
            best_total_loss = loss.detach().cpu().numpy()
            for param in model.parameters():
                best_weights = param.detach().cpu().numpy()[0].copy()
  
        loss.backward()
        optimizer.step()
        losses.append(loss)

    if plot_loss:
        fig2, ax2 = plt.subplots()
        ax2.plot(losses,color = "b")
        plt.plot([0,len(losses)], [min(losses), min(losses)], marker = 'o',color = "black")
        plt.show()

    if preference_assum == "regret":
        for param in model.parameters():
            param.data = nn.parameter.Parameter(torch.tensor(best_weights).double().to(device))
    train_loss = reward_pred_loss(model(X_train), y_train).detach().cpu().numpy()
    train_batch_size = y_train.size()[0]
    train_loss /= (train_batch_size)

    print ("training loss: " + str(train_loss))
   
    for param in model.parameters():
        reward_vector = param.detach().cpu().numpy()
    if len(reward_vector) == 1:
        reward_vector = reward_vector[0]
    print ("Learned reward weights:")
    print (reward_vector)

    return reward_vector,losses,train_loss



all_reward_vecs = []
all_train_losses = []
all_test_losses = []
all_training_acc= []
all_testing_acc = []
all_total_training_losses = []
all_avg_returns = []
#
# trajpairs = []
# phis = []


if use_random_MDPs_n_length_segs:
    num_near_opt = {3:0,6:0,9:0,12:0,15:0}
    num_above_random = {3:0,6:0,9:0,12:0,15:0}
    all_scaled_returns =  {3:[],6:[],9:[],12:[],15:[]}
    num_prefs = 3000
    for trial in range(400,430):
        all_states = None
        all_actions = None

        with open("random_MDPs/MDP_" + str(trial) +"all_trajss.npy","rb") as f:
            all_trajss = pickle.load(f)
        with open("random_MDPs/MDP_" + str(trial) +"all_sess.npy","rb") as f:
            all_sess = pickle.load(f)

        gt_rew_vec = np.load("random_MDPs/MDP_" + str(trial) +"gt_rew_vec.npy")

        succ_feats = np.load("random_MDPs/MDP_" + str(trial) +"succ_feats.npy")
        succ_q_feats = np.load("random_MDPs/MDP_" + str(trial) +"succ_q_feats.npy")

        with open("random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
            env = pickle.load(rf)

        seg_lengths = [3,6,9,12,15]
        for i in range(len(seg_lengths)):
            print ("=============== TRAJ LENGTH: " + str(seg_lengths[i]) + " ===============")
            
            all_trajs = all_trajss[i]
            all_ses = all_sess[i]
            if len(all_trajs)>num_prefs:
                np.random.seed(0)
                print ("SUBSAMPLING TRAJ PAIRS")
                idx = np.random.choice(np.arange(len(all_trajs)), num_prefs, replace=False)
                all_trajs = np.array(all_trajs)[idx]
                all_ses = np.array(all_ses)[idx]

           
            all_X = []
            all_r = [] 
            for i_, traj_pair in enumerate(all_trajs):
                
               
                phi_dis1,phi1 = find_reward_features(traj_pair[0],env,use_extended_SF=use_extended_SF,GAMMA=GAMMA,traj_length=seg_lengths[i])
                phi_dis2,phi2 = find_reward_features(traj_pair[1],env,use_extended_SF=use_extended_SF,GAMMA=GAMMA,traj_length=seg_lengths[i])
                all_r.append([np.dot(gt_rew_vec,phi_dis1[0:6]), np.dot(gt_rew_vec,phi_dis2[0:6])])
                all_X.append([phi_dis1, phi_dis2])

            pr_X,synth_max_y,expected_returns = generate_synthetic_prefs(all_X,all_r,all_ses,all_actions,all_states,mode,gt_rew_vec=np.array(gt_rew_vec),env=env)


            aX, ay = augment_data(pr_X,synth_max_y,"arr")
            rew_vect,all_losses,train_loss = train(aX, ay,plot_loss=False,gt_rew_vec=np.array(gt_rew_vec),env=env)

            print ("# of synthetic prefrences: " + str(len(pr_X)))
            print ("Ground truth reward vector: " + str(gt_rew_vec))
            V,Q = value_iteration(rew_vec =rew_vect,GAMMA=0.999,env=env)
            pi = build_pi(Q,env=env)
            V_under_gt = iterative_policy_evaluation(pi,rew_vec = np.array(gt_rew_vec), GAMMA=0.999,env=env)

            avg_return = np.sum(V_under_gt)/env.n_starts
            print ("average return following learned policy: ")
            print (avg_return)


            gt_avg_return = get_gt_avg_return(gt_rew_vec=np.array(gt_rew_vec), env=env, GAMMA=0.999)

            #build random policy
            random_pi = build_random_policy(env=env)
            print ("evaluating random policy under reward vec: " + str(gt_rew_vec))
            V_under_random_pi = iterative_policy_evaluation(random_pi,rew_vec = np.array(gt_rew_vec), GAMMA=0.999,env=env)
            random_avg_return = np.sum(V_under_random_pi)/env.n_starts
            print ("random policies avg return: ")
            print (random_avg_return)

            #scale everything: f(z) = (z-x) / (y-x)
            scaled_return = (avg_return - random_avg_return)/(gt_avg_return - random_avg_return)
            all_scaled_returns[seg_lengths[i]].append(scaled_return)

            print ("scaled return following learned policy: " + str(scaled_return))

            random_policy_data.changed_gt_rew_vec = False
            if (scaled_return >= 0.9):
                num_near_opt[seg_lengths[i]]+=1
            if (scaled_return >= 0):
                num_above_random[seg_lengths[i]] +=1

            print ("=================================================================================\n")

        np.save("multi_length_segs_res/400_430_num_prefs="+str(num_prefs)+"main_avg_return_" + str(mode) + "_" + str(preference_model) + "_" + str(preference_assum) + str(use_extended_SF) + "_" +  str(GAMMA) + ".npy",all_scaled_returns)
     
        for seg_length in seg_lengths:
            print ("=============== TRAJ LENGTH: " + str(seg_length) + " ===============")
            print ("% of MDPs where near optimal performance was achieved: " + str(100*(num_near_opt[seg_length]/30)) + "%")
            print ("% of MDPs where better than random performance was achieved: " + str(100*(num_above_random[seg_length]/30)) + "%")

elif use_random_MDPs:
    all_num_prefs = [3000]

    if run_temp_exp:
        for j in range(29):
            all_num_prefs.append(3000)

    num_near_opt = 0
    num_above_random = 0
    all_avg_returns =[]
    all_scaled_returns = []
    n_runs = 0

    for num_prefs in all_num_prefs:
        print ("============== NUM PREFS: " + str(num_prefs) + " ==============")
        
        for trial in range(0,30):
            np.random.seed(n_runs)
            n_runs+=1
        
            all_states = None
            all_actions = None
            all_trajs = np.load("random_MDPs/MDP_" + str(trial) +"all_trajs.npy",mmap_mode="r").tolist()
            all_ses = np.load("random_MDPs/MDP_" + str(trial) +"all_ses.npy",mmap_mode="r").tolist()
            gt_rew_vec = np.load("random_MDPs/MDP_" + str(trial) +"gt_rew_vec.npy",mmap_mode="r")

            if trial < 30 and GAMMA != 0.999:
                succ_feats = np.load("random_MDPs/MDP_" + str(trial) +"succ_feats_gamma="+str(GAMMA)+".npy",mmap_mode="r")
                
                if use_extended_SF:
                    sa_succ_feats = np.load("random_MDPs/MDP_" + str(trial) +"sa_succ_feats_gamma="+str(GAMMA)+".npy",mmap_mode="r")
                    succ_feats = np.concatenate((succ_feats, sa_succ_feats), axis=3)
                succ_q_feats = np.load("random_MDPs/MDP_" + str(trial) +"succ_q_feats_gamma="+str(GAMMA)+".npy",mmap_mode="r")
            else:
                succ_feats = np.load("random_MDPs/MDP_" + str(trial) +"succ_feats.npy",mmap_mode="r")
                succ_q_feats = np.load("random_MDPs/MDP_" + str(trial) +"succ_q_feats.npy",mmap_mode="r")

        
            with open("random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
                env = pickle.load(rf)
                if trial < 200 or trial > 300:
                    env.generate_transition_probs()
                    env.set_custom_reward_function(gt_rew_vec)

            if trial >307:
                all_traj_uniq = []
                all_ses_uniq = []
                for traj,ses in zip(all_trajs,all_ses):
                    for j in range (5):
                        all_traj_uniq.append(traj)
                        all_ses_uniq.append(ses)
                all_trajs = all_traj_uniq 
                all_ses = all_ses_uniq

            if len(all_trajs)>num_prefs and trial < 200:
                print ("SUBSAMPLING TRAJ PAIRS")
                idx = np.random.choice(np.arange(len(all_trajs)), num_prefs, replace=False)
                all_trajs = np.array(all_trajs)[idx]
                all_ses = np.array(all_ses)[idx]

            #only recalculate these for non-prob gridworlds
            if trial < 200 or trial > 300:
                all_X = []
                all_r = []

                # print (env.board)
                #TODO: CANNOT USE THIS find_reward_features METHOD WITH STOCHASTIC TRANSITIONS
                for i_, traj_pair in enumerate(all_trajs):
                    phi_dis1,phi1 = find_reward_features(traj_pair[0],env,use_extended_SF=use_extended_SF,GAMMA=GAMMA)
                    phi_dis2,phi2 = find_reward_features(traj_pair[1],env,use_extended_SF=use_extended_SF,GAMMA=GAMMA)
                    all_r.append([np.dot(gt_rew_vec,phi1[0:6]), np.dot(gt_rew_vec,phi2[0:6])])
                    all_X.append([phi1, phi2])

        
            pr_X,synth_max_y,expected_returns = generate_synthetic_prefs(all_X,all_r,all_ses,all_actions,all_states,mode,gt_rew_vec=np.array(gt_rew_vec),env=env)
            
        

            aX, ay = augment_data(pr_X,synth_max_y,"arr")
            
            rew_vect,all_losses,train_loss = train(aX, ay,plot_loss=False,gt_rew_vec=np.array(gt_rew_vec),env=env)
            
            if not run_temp_exp:
                np.save("discount_exp_res/0-30_trial=" + str(trial) + "_" + preference_model + "_" + preference_assum + "_rew_vec", rew_vect)

            print ("# of synthetic preferences: " + str(len(pr_X)))
            print ("Ground truth reward vector: " + str(gt_rew_vec))
            V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA,env=env)
            pi = build_pi(Q,env=env)
            V_under_gt = iterative_policy_evaluation(pi,rew_vec = np.array(gt_rew_vec), GAMMA=0.999,env=env)

            avg_return = np.sum(V_under_gt)/env.n_starts
            all_avg_returns.append(avg_return)
            print ("average return following learned policy: ")
            print (avg_return)


            gt_avg_return = get_gt_avg_return(gt_rew_vec=np.array(gt_rew_vec), env=env, GAMMA=0.999)

            #build random policy
            random_pi = build_random_policy(env=env)
            print ("evaluating random policy under reward vec: " + str(gt_rew_vec))
            V_under_random_pi = iterative_policy_evaluation(random_pi,rew_vec = np.array(gt_rew_vec), GAMMA=0.999,env=env)
            random_avg_return = np.sum(V_under_random_pi)/env.n_starts
            print ("random policies avg return: ")
            print (random_avg_return)

            #scale everything: f(z) = (z-x) / (y-x)
            scaled_return = (avg_return - random_avg_return)/(gt_avg_return - random_avg_return)
            all_scaled_returns.append(scaled_return)

            print ("scaled return following learned policy: " + str(scaled_return))

            random_policy_data.changed_gt_rew_vec = False
            if (scaled_return >= 0.9):
                num_near_opt+=1
            if (scaled_return >= 0):
                num_above_random+=1

            # assert False
            del all_trajs
            del all_ses
            del gt_rew_vec
            del succ_feats
            del succ_q_feats
            print ("=================================================================================\n")

        # np.save("discounting_expirements/100_200_num_prefs="+str(num_prefs)+"main_avg_return_" + str(mode) + "_" + str(preference_model) + "_" + str(preference_assum) + str(use_extended_SF) + "_" +  str(GAMMA) + ".npy",all_scaled_returns)
    print ("% of MDPs where near optimal performance was achieved: " + str(100*(num_near_opt/100)) + "%")
    print ("% of MDPs where better than random performance was achieved: " + str(100*(num_above_random/100)) + "%")

elif mode == "deterministic_user_data":
    if include_actions:
        assert False
    vf_X, vf_r, vf_y, vf_ses, vf_as, vf_states, pr_X, pr_r, pr_y, pr_ses, pr_as, pr_states, none_X, none_r, none_y, none_ses,none_as, none_states = get_all_statistics_aug_human()

    X_copy = none_X.copy()
    r_copy = none_r.copy()
    y_copy = none_y.copy()
    ses_copy = none_ses.copy()

    if partition_human_data:
        combined = list(zip(X_copy, r_copy, y_copy, ses_copy))
        random.Random(100).shuffle(combined)
        X_copy, r_copy, y_copy, ses_copy = zip(*combined)
        n=100

        X_copys = [X_copy[i::n] for i in range(n)]
        r_copys = [r_copy[i::n] for i in range(n)]
        y_copys = [y_copy[i::n] for i in range(n)]
        ses_copys = [ses_copy[i::n] for i in range(n)]
        avg_returns = []
        num_near_opt = 0
        for X_copy, r_copy,y_copy,ses_copy in zip(X_copys, r_copys,y_copys,ses_copys):
            print ("Number of human prefs used: " + str(len(X_copy)))


            X_copy, y_copy, X_copy_sytnh, y_copy_synth,_,loss_coef = clean_y(X_copy, r_copy,y_copy,ses_copy)
            aX, ay = augment_data(X_copy,y_copy,"arr")

            print ("finding reward vector...")
            rew_vect,all_losses,train_loss = train(aX, ay, loss_coef, plot_loss=False)
            print ("performing value iteration...")
            V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA)

            print ("following policy...")
            # avg_return = follow_policy(Q,100,viz_policy=False)
            pi = build_pi(Q)
            V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA)
            avg_return = np.sum(V_under_gt)/92 #number of possible start states
            print ("average return following learned policy: ")
            print (avg_return)
            print ("====================================================\n")
            avg_returns.append(avg_return)
            if avg_return > 30:
                num_near_opt+=1

        print ("% near optimal: " + str(num_near_opt/n))
        np.save(preference_model + "_" + preference_assum + "_avg_returns_n_split=" + str(n) + ".npy",avg_returns)

    else:
        print ("Number of human prefs used: " + str(len(X_copy)))
        X_copy, y_copy, X_copy_sytnh, y_copy_synth,_,loss_coef = clean_y(X_copy, r_copy,y_copy,ses_copy)
        aX, ay = augment_data(X_copy,y_copy,"arr")
        print ("finding reward vector...")
        rew_vect,all_losses,train_loss = train(aX, ay, loss_coef, plot_loss=False)
        print ("performing value iteration...")
        V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA)

        print ("following policy...")
        pi = build_pi(Q)
        V_under_gt = iterative_policy_evaluation(pi, GAMMA=GAMMA)
        avg_return = np.sum(V_under_gt)/92 #number of possible start states
        print ("average return following learned policy: ")
        print (avg_return)

elif mode == "sigmoid":
    vf_X, vf_r, vf_y, vf_ses, vf_as, vf_states, pr_X, pr_r, pr_y, pr_ses, pr_as, pr_states, none_X, none_r, none_y, none_ses,none_as, none_states = get_all_statistics_aug_human(include_dif_seg_lengths=include_dif_seg_lengths,GAMMA=GAMMA)

    pr_X = none_X
    pr_r = none_r
    pr_ses = none_ses
    pr_as = None
    pr_states = None

    pr_X_copy = pr_X.copy()
    pr_r_copy = pr_r.copy()
    pr_as_copy = None
    pr_states_copy = None

    print (len(none_X))

    for prob_iter in range(n_prob_iters):
        pr_X,synth_max_y,_ = generate_synthetic_prefs(pr_X_copy,pr_r_copy,pr_ses,pr_as_copy,pr_states_copy,mode)
        
        aX, ay = augment_data(pr_X,synth_max_y,"arr")

        print ("==========================Trial " + str(prob_iter)+" ==========================")
        print ("finding reward vector...")
        rew_vect,all_losses,train_loss = train(aX, ay,plot_loss=False)
        all_reward_vecs.append(rew_vect)
        all_train_losses.append(train_loss)
       
        all_total_training_losses.append(all_losses)

        print (rew_vect)

        print ("performing value iteration...")
        V,Q = value_iteration(rew_vec =rew_vect,GAMMA=GAMMA)

        print ("following policy...")
        # avg_return = follow_policy(Q,100,viz_policy=False)
        pi = build_pi(Q)
        V_under_gt = iterative_policy_evaluation(pi, GAMMA=0.999)
        avg_return = np.sum(V_under_gt)/92
        all_avg_returns.append(avg_return)
        print ("============================================================")

    print ("\n\n")
    print ("GAMMA: " + str(GAMMA))
    print ("Across all " + str(n_prob_iters) + " trials,")

    disp_mmv(all_avg_returns, "Average Return",None)
    get_gt_avg_return(GAMMA=0.999)
    disp_mmv(all_train_losses, "Training Loss",None)
    disp_mmv(all_reward_vecs, "Reward Vector",0)

    np.save("all_avg_returns_" + preference_model + "_" + preference_assum + str(keep_ties) + str(include_dif_seg_lengths) + ".npy", all_avg_returns)

else:
    vf_X, vf_r, vf_y, vf_ses, vf_as,vf_states, pr_X, pr_r, pr_y, pr_ses, pr_as, pr_states, none_X, none_r, none_y, none_ses, none_as, none_states = get_all_statistics_aug_human(include_dif_seg_lengths=include_dif_seg_lengths,GAMMA=GAMMA) #gamma used for labeling


    X,synth_max_y,expected_returns = generate_synthetic_prefs(none_X,none_r,none_ses,none_as,none_states,mode)

    print ("# of synthetic preferences: " + str(len(X)))
    aX, ay = augment_data(X,synth_max_y,"arr")

    print ("finding reward vector...")

    rew_vect,all_losses,train_loss = train(aX, ay,plot_loss=False)
    print ("performing value iteration...")
    V,Q = value_iteration(rew_vec =rew_vect,GAMMA=0.999)

    # print ("following policy...")
    # follow_policy(Q,100,viz_policy=True)
    print ("following policy...")
    pi = build_pi(Q)
    V_under_gt = iterative_policy_evaluation(pi, GAMMA=0.999)
    avg_return = np.sum(V_under_gt)/92
    print ("average return following learned policy: ")
    print (avg_return)

    get_gt_avg_return(GAMMA=0.999)
