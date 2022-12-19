from grid_world import GridWorldEnv
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_random_reward_vector():
    space = [-1,50,-50,1,-1,-2]
    vector = []
    for i in range(6):
        s = random.choice(space)
        space.remove(s)
        vector.append(s)
    return vector

def learn_successor_feature_iter(pi,FGAMMA,rew_vec=None,env=None):
    if env == None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    if type(rew_vec) is np.ndarray:
        env.set_custom_reward_function(rew_vec[0:6])

    THETA = 0.001
    # initialize Q
    # Q = defaultdict(lambda: np.zeros(env.action_space))
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    psi = [[np.zeros(env.feature_size) for i in range(env.width)] for j in range (env.height)]
    psi_actions = [[np.zeros(len(actions) * env.width * env.height) for i in range(env.width)] for j in range (env.height)]
    psi_Q = [[[np.zeros(env.feature_size) for a in range(len(actions))] for i in range(env.width)] for j in range (env.height)]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state value
    #----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        delta = 0
        # new_psi = psi.copy()
        new_psi = np.copy(psi)

        for i in range (env.height):
            for j in range (env.width):
                if env.is_blocked(i,j):
                    continue
                # total = 0

                state_psi = []
                action_psi = []
                for trans in pi[(i,j)]:
                    prob, a_index = trans
                    next_state, reward, done, phi = env.get_next_state((i,j),a_index)
                    
                    action_phi = np.zeros((env.height,env.width,len(actions)))
                    action_phi[i][j][a_index] = 1
                    action_phi = np.ravel(action_phi)
                    
                    ni,nj = next_state
                    if not done:
                        psi_sas = prob*(phi + FGAMMA*psi[ni][nj])
                        psi_q = (phi + FGAMMA*psi[ni][nj])
                        action_feat = prob*(action_phi + FGAMMA*psi_actions[ni][nj])
                    else:
                        psi_sas = np.zeros(env.feature_size)
                        psi_q = np.zeros(env.feature_size)
                        action_feat = np.zeros(len(actions) * env.width * env.height)
                    
                    psi_Q[i][j][a_index] = psi_q
                    state_psi.append(psi_sas)
                    action_psi.append(action_feat)

                psi_actions[i][j] = sum(action_psi)
                new_psi[i][j] = sum(state_psi)
                delta = max(delta,np.sum(np.abs(psi[i][j]-new_psi[i][j])))
                # print (np.sum(np.abs(psi[i][j]-new_psi[i][j])))

        psi = new_psi

        if delta < THETA:
            break
    return psi,np.array(psi_actions), psi_Q


def build_pi_from_nn_feats(model, env=None):
    pi = {}
    if env == None:
        height = 10
        width = 10
    else:
        height = env.height
        width = env.width
    for i in range (height):
        for j in range (width):
            with torch.no_grad():
                # greedy_a_i = np.argmax([model.get_trans_val(torch.tensor([i,j,a_i]).to(device).float()).cpu() for a_i in range(4)])
                max_weight = np.max([model.get_trans_val(torch.tensor([i,j,a_i]).to(device).float()).cpu() for a_i in range(4)])
                count = [model.get_trans_val(torch.tensor([i,j,a_i]).to(device).float()).cpu() for a_i in range(4)].count(max_weight)
                pi[(i,j)] = [(1/count if model.get_trans_val(torch.tensor([i,j,a_index]).to(device).float()).cpu()  == max_weight else 0, a_index) for a_index in range(4)]

            #pi[(i,j)] = [(1 if a_index == greedy_a_i else 0, a_index) for a_index in range(4)]

    return pi


def build_pi_from_feats(s_a_weights,env=None):
    pi = {}
    if env == None:
        height = 10
        width = 10
    else:
        height = env.height
        width = env.width

    s_a_weights = s_a_weights.reshape((env.height,env.width,4))

    for i in range (height):
        for j in range (width):
            # greedy_a_i = np.argmax(s_a_weights[i][j])
            max_weight = max(s_a_weights[i][j])
            count = s_a_weights[i][j].tolist().count(max_weight)
            pi[(i,j)] = [(1/count if s_a_weights[i][j][a_index] == max_weight else 0, a_index) for a_index in range(4)]

    return pi

def build_pi(Q,env=None):
    pi = {}
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    if env == None:
        height = 10
        width = 10
    else:
        height = env.height
        width = env.width

    for i in range (height):
        for j in range (width):
            V = max(Q[i][j])
            V_count = Q[i][j].tolist().count(V)
            pi[(i,j)] = [(1/V_count if Q[i][j][a_index] == V else 0, a_index) for a_index in range(len(actions))]
    return pi


def build_random_policy(env=None):
    pi = {}
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    if env == None:
        height = 10
        width = 10
    else:
        height = env.height
        width = env.width

    for i in range (height):
        for j in range (width):
            pi[(i,j)] = [(1/len(actions),a_index) for a_index in range(len(actions))]
    return pi

def iterative_policy_evaluation(pi,rew_vec=None, set_rand_rew = False, GAMMA=0.999, env=None):
    if env == None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    # rand_rew_vec = get_random_reward_vector()
    # rand_rew_vec = [-1, -50, 50, 1, -1, -2]
    if  type(rew_vec) is np.ndarray:
        env.set_custom_reward_function(rew_vec[0:6])
    elif set_rand_rew:
        rand_rew_vec = get_random_reward_vector()
        env.set_custom_reward_function(rew_vec)

    THETA = 0.001
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    V = np.zeros((env.height, env.width))

    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state value
    #----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        delta = 0
        new_V = V.copy()
        for i in range (env.height):
            for j in range (env.width):
                if env.is_blocked(i,j):
                    continue
                # total = 0
                state_qs = []
                for trans in pi[(i,j)]:
                    prob, a_index = trans
                    next_state, reward, done, _ = env.get_next_state((i,j),a_index)
                    ni,nj = next_state
                    if not done:
                        state_qs.append(prob*(reward + GAMMA*V[ni][nj]))
                    else:
                        state_qs.append(prob*reward)
                new_V[i][j] = sum(state_qs)
                delta = max(delta,np.abs(V[i][j]-new_V[i][j]))

        V = new_V
        if delta < THETA:
            break

    return V


def value_iteration(rew_vec=None, set_rand_rew = False, GAMMA=0.999,env=None,is_set=False):
    # env = GridworldEnv()
    if env == None:
        env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
    # rand_rew_vec = get_random_reward_vector()
    # rand_rew_vec = [-1, -50, 50, 1, -1, -2]
    if  type(rew_vec) is np.ndarray and not is_set:
        env.set_custom_reward_function(rew_vec[0:6])
    elif set_rand_rew:
        rand_rew_vec = get_random_reward_vector()
        env.set_custom_reward_function(rew_vec)


    THETA = 0.001
    # initialize Q
    # Q = defaultdict(lambda: np.zeros(env.action_space))
    V = np.zeros((env.height, env.width))
    Qs = [[np.zeros(4) for i in range(env.width)] for j in range (env.height)]

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    n = 0

    #----------------------------------------------------------------------------------------------------------------------------------------
    #iterativley learn state value
    #----------------------------------------------------------------------------------------------------------------------------------------
    while True:
        delta = 0
        new_V = V.copy()
        for i in range (env.height):
            for j in range (env.width):
                if env.is_blocked(i,j):
                    continue
                v = V[i][j]
                state_Qs = []
                for a_index in range(len(actions)):

                    next_state, reward, done, _ = env.get_next_state((i,j),a_index)
                 
                    ni,nj = next_state
                    if not done:
                        Q = reward + GAMMA*V[ni][nj]
                    else:
                        Q = reward
                    Qs[i][j][a_index] = Q

                new_V[i][j] = max(Qs[i][j])
                delta = max(delta,np.abs(v-new_V[i][j]))

        V = new_V
        if delta < THETA:
            break
    # print (np.round(V,1))
    # print ("=================================================================================\n")
    return V,Qs


def get_gt_avg_return(GAMMA,gt_rew_vec=None,env=None):
    V,Q = value_iteration(rew_vec=gt_rew_vec, env=env,GAMMA=GAMMA)
    pi = build_pi(Q,env=env)
    V_under_gt = iterative_policy_evaluation(pi, rew_vec=gt_rew_vec, env=env, GAMMA=0.999)
    if env == None:
        n_starts = 92
    else:
        n_starts = env.n_starts
    gt_avg_return = np.sum(V_under_gt/n_starts)
    # print ("average return following ground truth policy: ")
    # print (gt_avg_return)
    # print ("=================================================================================\n")
    return gt_avg_return
