import numpy as np
import random
import pickle
import itertools
from itertools import permutations
from itertools import combinations
import math

from grid_world import GridWorldEnv
from rl_algos import value_iteration
from generate_random_policies import generate_all_policies, calc_value


def randomly_place_item_exact(env,id,N,height,width):
    #randomly place mud
    for i in range(N):
        x = random.randint(0,height-1)
        y = random.randint(0,width-1)
        while env.board[x][y] != 0 and env.board[x][y] != 6:
            x = random.randint(0,height-1)
            y = random.randint(0,width-1)

        env.board[x][y] = id
        # if env.board[x][y] == 6 and 6+id <= 11:
        #     env.board[x][y] = id+6
        # else:
        #     env.board[x][y] = id


def contains_cords(arr1,arr2):
    for a in arr1:
        if a[0] == arr2[0] and a[1] == arr2[1]:
            return True
    return False


def is_in_gated_area(x,y,board):
    val = board[x][y]
    if  val >= 6:
        return True
    else:
        return False

def is_in_blocked_area(x,y,board):
    val = board[x][y]
    if val == 2 or val == 8:
        return True
    else:
        return False

def find_end_state(traj,board):
    in_gated =False
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    if is_in_gated_area(traj_ts_x,traj_ts_y,board):
        in_gated = True

    for i in range (1,4):
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(env.board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(env.board[0]):
            if not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1],env.board):
                next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1],env.board)
                if in_gated == False or  (in_gated and next_in_gated):
                    traj_ts_x += traj[i][0]
                    traj_ts_y += traj[i][1]
    return traj_ts_x, traj_ts_y


def get_state_feature(env,x,y):
    reward_feature = np.zeros(6)
    if env.board[x][y] == 0:
        reward_feature[0] = 1
    elif env.board[x][y] == 1:
        #flag
        # reward_feature[0] = 1
        reward_feature[1] = 1
    elif env.board[x][y] == 2:
        #house
        # reward_feature[0] = 1
        pass
    elif env.board[x][y] == 3:
        #sheep
        # reward_feature[0] = 1
        reward_feature[2] = 1
    elif env.board[x][y] == 4:
        #coin
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[3] = 1
    elif env.board[x][y] == 5:
        #road block
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[4] = 1
    elif env.board[x][y] == 6:
        #mud area
        # reward_feature[0] = 1
        reward_feature[5] = 1
    elif env.board[x][y] == 7:
        #mud area + flag
        reward_feature[1] = 1
    elif env.board[x][y] == 8:
        #mud area + house
        pass
    elif env.board[x][y] == 9:
        #mud area + sheep
        reward_feature[2] = 1
    elif env.board[x][y] == 10:
        #mud area + coin
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[3] = 1
    elif env.board[x][y] == 11:
        #mud area + roadblock
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[4] = 1
    return reward_feature


def find_reward_features(traj,env,traj_length=3):
    GAMMA=1
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    phi = np.zeros(6)
    phi_dis = np.zeros(6)

    for i in range (1,traj_length+1):
        # print ("===========================")
        # print (traj_ts_x,traj_ts_y)
        # print ("===========================")
        #check if we are at terminal state
        if env.board[prev_x, prev_y] == 1 or env.board[traj_ts_x, traj_ts_y] == 3 or env.board[traj_ts_x, traj_ts_y] == 7 or env.board[traj_ts_x, traj_ts_y] == 9:
            continue
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(env.board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(env.board[0]) and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1], env.board):
            # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
            # if in_gated == False or  (in_gated and next_in_gated):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]
        if (traj_ts_x,traj_ts_y) != (prev_x,prev_y):
            # print ("state feature: ")
            # print (get_state_feature(traj_ts_x,traj_ts_y,env))
            dis_state_sf = (GAMMA**(i-1))*get_state_feature(env,traj_ts_x,traj_ts_y)
            state_sf = get_state_feature(env,traj_ts_x,traj_ts_y)
        else:
            #check if we are at terminal state
            if env.board[prev_x, prev_y] == 1 or env.board[traj_ts_x, traj_ts_y] == 3 or env.board[traj_ts_x, traj_ts_y] == 7 or env.board[traj_ts_x, traj_ts_y] == 9:
                dis_state_sf = [0,0,0,0,0,0]
                state_sf = [0,0,0,0,0,0]
            else:
                dis_state_sf = (GAMMA**(i-1))*(get_state_feature(env,traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])
                state_sf = (get_state_feature(env,traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])
        phi_dis += dis_state_sf
        phi+= state_sf

        prev_x = traj_ts_x
        prev_y = traj_ts_y
    # print ("--done--\n")
    return phi_dis,phi

def create_traj_prob(s0_x, s0_y, action_seq,traj_length, env, values):
    traj = [(s0_x,s0_y)]
    states = [(s0_x,s0_y)]
    x,y = (s0_x,s0_y)
    t_partial_r_sum = 0
    reward_feature = np.zeros(env.feature_size)
    for i, action in enumerate(action_seq):

        traj.append(action)
        a_i = find_action_index(action)
        next_state, reward, done, phi = env.get_next_state_prob((x,y), a_i)

        assert reward == np.dot(phi, env.reward_array)

        reward_feature += phi

        states.append(next_state)

        if i < traj_length:
            x,y = next_state

        t_partial_r_sum+= reward
        is_terminal = done
    # print (t_partial_r_sum)
    # print (np.dot(reward_feature, env.reward_array))
    # print (reward_feature)
    # print (env.reward_array)

    # print ("\n")
    assert t_partial_r_sum == np.dot(reward_feature, env.reward_array)
    return traj, t_partial_r_sum, values[x][y],is_terminal, x, y, states, reward_feature


def create_traj(s0_x, s0_y,action_seq,traj_length, board, rewards_function,terminal_states,blocking_cords,values):
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #generate trajectory 1
    # t_partial_r_sum = rewards_model[s0_x][s0_y]
    t_partial_r_sum = 0

    traj = [(s0_x,s0_y)]
    x = s0_x
    y = s0_y
    step_n = 0
    is_terminal = False
    states = [(s0_x,s0_y)]
    # print ("start: " + str((x,y)))
    for step1 in range(traj_length+1):
        if contains_cords(terminal_states,[x,y]):
            # if step_n != traj_length:
            #     return False, False, False, False, False, False
            # else:
            is_terminal = True

        if step_n != traj_length:
            a = action_seq[step1]

            traj.append(a)
            a_i = find_action_index(a)
            # print (((x,y),rewards_function[x][y][a_i]))
            t_partial_r_sum += rewards_function[x][y][a_i]
            #and not (contains_cords(oneway_cords, [x,y]) and not contains_cords(oneway_cords, [x+a[0],y+a[1]]))
            if (x + a[0] >= 0 and x + a[0] < len(board) and y + a[1] >= 0 and y + a[1] < len(board[0])) and not contains_cords(blocking_cords,[x + a[0],  y + a[1]]) and not contains_cords(terminal_states,[x + a[0],  y + a[1]]):
                x = x + a[0]
                y = y + a[1]
            states.append((x,y))
        # else:
        #     # print ("end state: " + str(((x,y))))
        #     traj.append(((x,y),[0,0]))

        step_n+=1

    return traj, t_partial_r_sum, values[x][y],is_terminal, x, y, states


def check_same_policy(prev_Q, curr_Q,env):

    for x in range(env.height):
        for y in range(env.width):
            if np.argmax(prev_Q[x][y]) != np.argmax(curr_Q[x][y]):
                # print (prev_Q[x][y])
                # print (curr_Q[x][y])
                return False


def generate_stoch_MDP(r_win):
    height = 5
    width = 5

    env = GridWorldEnv(None, height, width)

    #place sheep
    sheep_props = [0.05, 0.1, 0.3]
    sheep_prop = random.choice(sheep_props)
    n_sheep = int(sheep_prop*height*width)
    randomly_place_item_exact(env,3,n_sheep,height,width)

    #place goal
    x = random.randint(0,height-1)
    y = random.randint(0,width-1)
    env.board[x][y] = 1


    # env.board[0][0] = 3
    # env.board[1][0] = 0
    # env.board[2][0] = 1
    #
    rew_vec = [-1,50,-50,r_win,0,0]
    env.set_custom_reward_function(rew_vec,set_global=True)
    env.find_n_starts()

    env.get_blocking_cords()
    env.get_terminal_cords()

    sheep_trans_prob = 0.5

    for x in range(env.height):
        for y in range(env.width):
            for a_index in range(4):
                _, _, _, phi = env.get_next_state((x,y),a_index)

                orig_trans = list(env.transition_probs[x][y][a_index].keys())
                if len(orig_trans) !=1:
                    continue
                next_state, reward, done, reward_feature = orig_trans[0]

                if phi[2] == 1:
                    #found sheep
                    # assert done==True
                    env.transition_probs[x][y][a_index] ={(next_state, reward, done, reward_feature):sheep_trans_prob, (next_state, rew_vec[3], done, tuple([0,0,0,1,0,0])):1-sheep_trans_prob}

    all_X, all_r, all_ses, all_trajs, all_states, all_actions = subsample_env_trajs(env)
    all_env_boards = None
    return env, all_X, all_r, all_ses, all_trajs, all_env_boards, all_states, all_actions

def generate_MDP(prob=False,n_length_trajs=False):
    
    
    dimensions_width = [5,6,10]
    dimensions_height = [3,6,10,15]
    # dimensions_width = [1]
    # dimensions_height = [3]

    # dimensions_width = [10]
    # dimensions_height = [10]
    height = random.choice(dimensions_height)
    width = random.choice(dimensions_width)

    env = GridWorldEnv(None, height, width)


    #mud
    # mud_props = [0, 0.3, 0.6]
    # mud_prop = random.choice(mud_props)
    # n_mud = int(mud_prop*height*width)
    # randomly_place_item_exact(env,6,n_mud,height,width)


    #sheep
    sheep_props = [0, 0.1, 0.3]
    # sheep_props = [0]

    sheep_prop = random.choice(sheep_props)
    n_sheep = int(sheep_prop*height*width)
    randomly_place_item_exact(env,3,n_sheep,height,width)


    #mildly bad
    mildly_bad_props = [0, 0.1, 0.5, 0.8]
    # mildly_bad_props = [0]

    mildly_bad_prop = random.choice(mildly_bad_props)
    while (mildly_bad_prop + sheep_prop >= 1):
        mildly_bad_prop = random.choice(mildly_bad_props)
    n_mildly_bad = int(mildly_bad_prop*height*width)
    randomly_place_item_exact(env,5,n_mildly_bad,height,width)

    #mildly good
    mildly_good_props = [0, 0.1, 0.2]
    # mildly_good_props = [0.8]
    mildly_good_prop = random.choice(mildly_good_props)
    while (mildly_bad_prop + sheep_prop + mildly_good_prop >= 1):
        mildly_good_prop = random.choice(mildly_good_props)
    n_mildly_good = int(mildly_good_prop*height*width)
    randomly_place_item_exact(env,4,n_mildly_good,height,width)

    #goal
    x = random.randint(0,height-1)
    y = random.randint(0,width-1)
    env.board[x][y] = 1

    goal_rews = [0, 1, 5, 10, 50]
    sheep_rews = [-5, -10, -50,-100]
    mildly_bad_rews = [-2, -5, -10]
    mildly_good_rews = [1]
    mud_rews = [-1,-2,-3]
    rew_vec = [-1,random.choice(goal_rews),random.choice(sheep_rews),random.choice(mildly_good_rews),random.choice(mildly_bad_rews),random.choice(mud_rews)]
    env.set_custom_reward_function(rew_vec,set_global=True)
    env.find_n_starts()

    env.get_blocking_cords()
    env.get_terminal_cords()
    V,Qs = value_iteration(rew_vec = np.array(rew_vec),GAMMA=0.999,env=env)

    if prob:
        V,Qs = value_iteration(rew_vec = np.array(env.reward_array),GAMMA=0.999,env=env,is_set=True)
        prev_Qs = Qs.copy()

        #get worst state
        worst_state = None
        worst_val = float("inf")
        for x in range(env.height):
            for y in range(env.width):
                if V[x][y] < worst_val and not env.is_terminal(x,y) and not env.is_blocked(x,y):
                    worst_val = V[x][y]
                    worst_state = (x,y)


        #create states that randomly jump to goal
        good_tele_prop = 0.75
        good_tele_prob = 0.9
        n_tele_sas = int(good_tele_prop*env.height*env.width*4)
        sa_teles = []

        #choose random (s,a) pairs to add teleportation
        for i in range(n_tele_sas):
            x = random.randint(0, env.height-1)
            y = random.randint(0, env.width-1)
            a = random.randint(0, 3)
            while (x,y,a) in sa_teles:
                x = random.randint(0, env.height-1)
                y = random.randint(0, env.width-1)
                a = random.randint(0, 3)
            sa_teles.append((x,y,a))
        #change selected (s,a) pairs and ensure that the optimal policy has not been changed
        n_failed = 0
        for sa in sa_teles:
            x,y,a = sa
            n_tries = 0
            if np.argmax(Qs[x][y]) == a:
                n_failed+=1
                continue

            #TODO: ASSUMES THAT TRANSITION PROB FOR (S,A) IS DETERMINISTIC AND HAS NOT BEEN MODIFIED YET
            orig_trans = list(env.transition_probs[x][y][a].keys())
            assert len(orig_trans) ==1
            next_state, reward, done, reward_feature = orig_trans[0]
            env.transition_probs[x][y][a] ={(next_state, reward, done, reward_feature):good_tele_prob, (env.get_goal_rand(), rew_vec[1], True, tuple([0,1,0,0,0,0])):1-good_tele_prob}

            # print (env.transition_probs[x][y][a])
            V,Qs = value_iteration(rew_vec = np.array(env.reward_array),GAMMA=0.999,env=env,is_set=True)

            changed_opt_policy = check_same_policy(prev_Qs, Qs,env)
            if changed_opt_policy:
                env.transition_probs[x][y][a] ={(next_state, reward, done, reward_feature):1}
                n_failed+=1



        coin_trans_prob = 0.95
        goal_trans_prob = 0.99
        #create prob transitions for coins and goal
        for x in range(env.height):
            for y in range(env.width):
                for a_index in range(4):
                    _, _, _, phi = env.get_next_state((x,y),a_index)

                    orig_trans = list(env.transition_probs[x][y][a_index].keys())
                    if len(orig_trans) !=1:
                        continue
                    next_state, reward, done, reward_feature = orig_trans[0]

                    if phi[3] == 1:
                        #found a coin
                        env.transition_probs[x][y][a_index] ={(next_state, reward, done, reward_feature):coin_trans_prob, (next_state, 2*rew_vec[4] + rew_vec[0], done, tuple([1,0,0,0,2,0])):1-coin_trans_prob}
                    elif phi[1] == 1:
                        #found goal
                        # assert done==True
                        env.transition_probs[x][y][a_index] ={(next_state, reward, done, reward_feature):goal_trans_prob, (worst_state, rew_vec[0], False, tuple([1,0,0,0,0,0])):1-goal_trans_prob}


        V,Qs = value_iteration(rew_vec = np.array(env.reward_array),GAMMA=0.999,env=env,is_set=True)
        changed_opt_policy = check_same_policy(prev_Qs, Qs,env)
        if changed_opt_policy:
            assert False


    if prob:
        all_X, all_r, all_ses, all_trajs = subsample_env_trajs(env)
        all_env_boards = None
    elif n_length_trajs:
        #collect N randomly samples segment pairs
        all_Xs = []
        all_rs = []
        all_sess = []
        all_trajss = []
        for traj_length in [3,6,12,15]:
            all_X, all_r, all_ses, all_trajs,_,_ = subsample_env_n_length_trajs(env,traj_length=traj_length)
            all_Xs.append(all_X)
            all_rs.append(all_r)
            all_sess.append(all_ses)
            all_trajss.append(all_trajs)
        return env, all_Xs, all_rs, all_sess, all_trajss,
    else:
        #collect N randomly samples segment pairs
        all_X, all_r, all_ses, all_trajs,_ = get_env_trajs(env)

    return env, all_X, all_r, all_ses, all_trajs


def decode(i):
    k = math.floor((1+math.sqrt(1+8*i))/2)
    return k,i-k*(k-1)//2

def rand_pair(n):
    return decode(random.randrange(n*(n-1)//2))

def rand_pairs(n,m):
    #https://stackoverflow.com/questions/55244113/python-get-random-unique-n-pairs
    return [decode(i) for i in random.sample(range(n*(n-1)//2),m)]

def find_action_index(action):
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    i = 0
    for a in actions:
        if a[0] == action[0] and a[1] == action[1]:
            return i
        i+=1
    return False

def get_action_indices(actions):
    indices = []
    for action in actions:
        indices.append(find_action_index(action))
    return indices

def get_traj_states(traj,board,traj_length=3):

    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]

    states = [[traj_ts_x, traj_ts_y]]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    prev_x = traj_ts_x
    prev_y = traj_ts_y
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    for i in range (1,traj_length+1):
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(board[0]) and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1],board):
            # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
            # if in_gated == False or  (in_gated and next_in_gated):
            if not(board[traj_ts_x][traj_ts_y] == 1 or board[traj_ts_x][traj_ts_y] == 3 or board[traj_ts_x][traj_ts_y] == 7 or board[traj_ts_x][traj_ts_y] == 9):
                traj_ts_x += traj[i][0]
                traj_ts_y += traj[i][1]

        states.append([traj_ts_x,traj_ts_y])

        prev_x = traj_ts_x
        prev_y = traj_ts_y
    # print ("--done--\n")
    return states


def subsample_env_trajs(env,traj_length):
    #6,9,15,21
    if traj_length == 3:
        all_action_seqs = list(itertools.product(env.actions,env.actions,env.actions))
    else:
        assert False
    # elif traj_length == 6:
    #     all_action_seqs = list(itertools.product(env.actions,env.actions,env.actions, env.actions,env.actions,env.actions))
    # elif traj_length == 9:
    #     all_action_seqs = list(itertools.product(env.actions,env.actions,env.actions, env.actions,env.actions,env.actions,env.actions,env.actions,env.actions))
    # elif traj_length == 12:
    #     all_action_seqs = list(itertools.product(env.actions,env.actions,env.actions, env.actions,env.actions,env.actions, env.actions,env.actions,env.actions,env.actions,env.actions,env.actions))
    # elif traj_length == 15:
    #     all_action_seqs = list(itertools.product(env.actions,env.actions,env.actions, env.actions,env.actions,env.actions, env.actions,env.actions,env.actions,env.actions,env.actions,env.actions))
    # elif traj_length == 21:
    #     all_action_seqs = list(itertools.product(env.actions,env.actions,env.actions, env.actions,env.actions,env.actions, env.actions,env.actions,env.actions, env.actions,env.actions,env.actions, env.actions,env.actions,env.actions,env.actions,env.actions,env.actions))

    V,Qs = value_iteration(rew_vec = np.array(env.reward_array),GAMMA=0.999,env=env,is_set=True)
    MAX_SAMPLES = 30000
    N_FAILS = 0
    N_SAMPLES = 0

    collected_traj_data = []

    while True:
        if N_SAMPLES > MAX_SAMPLES:
            break
        if N_FAILS >= 5*MAX_SAMPLES:
            break
        for x in range(env.height):
            for y in range(env.width):
                for a_seq in range(5):
                    action_seq_1 = random.choice(all_action_seqs)
                    t1_s0_x,t1_s0_y = (x,y)
                    traj1, t1_partial_r_sum, v_t1,is_terminal1,traj1_ts_x,traj1_ts_y, states1,phi1 = create_traj_prob(t1_s0_x, t1_s0_y,action_seq_1,traj_length, env, V)

                    traj1_ses = [(traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y)]

                    v_dif1 = v_t1 - V[t1_s0_x][t1_s0_y]

                    if (tuple(phi1), tuple(traj1), tuple(traj1_ses), t1_partial_r_sum, v_dif1) not in collected_traj_data:
                        collected_traj_data.append((tuple(phi1), tuple(traj1), tuple(traj1_ses), t1_partial_r_sum, v_dif1))
                        N_SAMPLES+=1
                    else:
                        N_FAILS+=1



    if len(collected_traj_data)*100 > len(collected_traj_data)*(len(collected_traj_data)-1)//2:
        pairs_indices = rand_pairs(len(collected_traj_data),len(collected_traj_data)*(len(collected_traj_data)-1)//2)
    else:
        pairs_indices = rand_pairs(len(collected_traj_data),len(collected_traj_data)*100)

    all_X = []
    all_r = []
    all_ses = []
    all_trajs = []
    all_states = []
    all_actions = []


    for pair_i in pairs_indices:
        p1_i, p2_i = pair_i

        phi1, traj1, traj1_ses, t1_partial_r_sum, v_dif1 = collected_traj_data[p1_i]
        phi1 = list(phi1)
        traj1 = list(traj1)
        traj1_ses = list(traj1_ses)

        phi2, traj2, traj2_ses, t2_partial_r_sum, v_dif2 = collected_traj_data[p2_i]
        phi2 = list(phi2)
        traj2 = list(traj2)
        traj2_ses = list(traj2_ses)



        partial_sum_dif = float(t2_partial_r_sum - t1_partial_r_sum)
        v_dif = float(v_dif2 - v_dif1)


        all_actions.append([get_action_indices(traj1[1:]),get_action_indices(traj2[1:])])
        all_states.append([get_traj_states(traj1,env.board,len(traj1)-1), get_traj_states(traj2,env.board,len(traj2)-1)])


        all_X.append([phi1, phi2])
        all_r.append([t1_partial_r_sum, t2_partial_r_sum])
        all_ses.append([traj1_ses,traj2_ses])
        all_trajs.append([traj1,traj2])

    if len(all_X) > 30000:
        idx = np.random.choice(np.arange(len(all_X)), 30000, replace=False)
        return np.array(all_X)[idx], np.array(all_r)[idx], np.array(all_ses)[idx], np.array(all_trajs)[idx], np.array(all_states)[idx], np.array(all_actions)[idx]
    else:
        return np.array(all_X), np.array(all_r), np.array(all_ses), np.array(all_trajs), np.array(all_states), np.array(all_actions)


def subsample_env_n_length_trajs(env,traj_length):
    V,Qs = value_iteration(rew_vec = np.array(env.reward_array),GAMMA=0.999,env=env,is_set=True)
    MAX_SAMPLES = 30000
    N_FAILS = 0
    N_SAMPLES = 0

    collected_traj_data = []

    while True:
        if N_SAMPLES > MAX_SAMPLES:
            break
        if N_FAILS >= 5*MAX_SAMPLES:
            break
        for x in range(env.height):
            for y in range(env.width):
                for a_seq in range(5):
                    action_seq_1 = [random.choice(env.actions) for i in range(traj_length)]
                    t1_s0_x,t1_s0_y = (x,y)
                    traj1, t1_partial_r_sum, v_t1,is_terminal1,traj1_ts_x,traj1_ts_y, states1,phi1 = create_traj_prob(t1_s0_x, t1_s0_y,action_seq_1,traj_length, env, V)

                    traj1_ses = [(traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y)]

                    v_dif1 = v_t1 - V[t1_s0_x][t1_s0_y]

                    if (tuple(phi1), tuple(traj1), tuple(traj1_ses), t1_partial_r_sum, v_dif1) not in collected_traj_data:
                        collected_traj_data.append((tuple(phi1), tuple(traj1), tuple(traj1_ses), t1_partial_r_sum, v_dif1))
                        N_SAMPLES+=1
                    else:
                        N_FAILS+=1



    if len(collected_traj_data)*100 > len(collected_traj_data)*(len(collected_traj_data)-1)//2:
        pairs_indices = rand_pairs(len(collected_traj_data),len(collected_traj_data)*(len(collected_traj_data)-1)//2)
    else:
        pairs_indices = rand_pairs(len(collected_traj_data),len(collected_traj_data)*100)

    all_X = []
    all_r = []
    all_ses = []
    all_trajs = []
    all_states = []
    all_actions = []


    for pair_i in pairs_indices:
        p1_i, p2_i = pair_i

        phi1, traj1, traj1_ses, t1_partial_r_sum, v_dif1 = collected_traj_data[p1_i]
        phi1 = list(phi1)
        traj1 = list(traj1)
        traj1_ses = list(traj1_ses)

        phi2, traj2, traj2_ses, t2_partial_r_sum, v_dif2 = collected_traj_data[p2_i]
        phi2 = list(phi2)
        traj2 = list(traj2)
        traj2_ses = list(traj2_ses)



        partial_sum_dif = float(t2_partial_r_sum - t1_partial_r_sum)
        v_dif = float(v_dif2 - v_dif1)


        all_actions.append([get_action_indices(traj1[1:]),get_action_indices(traj2[1:])])
        all_states.append([get_traj_states(traj1,env.board,len(traj1)-1), get_traj_states(traj2,env.board,len(traj2)-1)])


        all_X.append([phi1, phi2])
        all_r.append([t1_partial_r_sum, t2_partial_r_sum])
        all_ses.append([traj1_ses,traj2_ses])
        all_trajs.append([traj1,traj2])

    if len(all_X) > 30000:
        idx = np.random.choice(np.arange(len(all_X)), 30000, replace=False)
        return np.array(all_X)[idx], np.array(all_r)[idx], np.array(all_ses)[idx], np.array(all_trajs)[idx], np.array(all_states)[idx], np.array(all_actions)[idx]
    else:
        return np.array(all_X), np.array(all_r), np.array(all_ses), np.array(all_trajs), np.array(all_states), np.array(all_actions)



def get_env_trajs(env,traj_length=3):
    V,Qs = value_iteration(rew_vec = np.array(env.reward_array),GAMMA=0.999,env=env)
    print ("finding start states...")
    start_states = []
    for i in range(len(env.board)):
        for j in range(len(env.board[0])):
            if not contains_cords(env.terminal_cords,(i,j)) and not contains_cords(env.blocking_cords,(i,j)):


                for i2 in range(len(env.board)):
                    for j2 in range(len(env.board[0])):
                        if not contains_cords(env.terminal_cords,(i2,j2)) and not contains_cords(env.blocking_cords,(i2,j2)):
                            start_states.append([(i,j), (i2,j2)])
            # else:
            #     print (contains_cords(env.terminal_cords,(i,j)))
            #     print (contains_cords(env.blocking_cords,(i,j)))
            #     print ((i,j))

    # print (start_states)
    # print ("*************************************")
    all_action_seqs = list(combinations(itertools.product(env.actions,env.actions,env.actions),2))



    all_collected_trajs = []
    n_pairs_found = 0
    max_pairs = 10000-1

    all_X = []
    all_r = []
    all_ses = []
    all_trajs = []
    all_env_boards = []

    seen_traj_pairs = set()

    for start_state in start_states:
        # print ("======")
        # print (len(all_action_seqs))
        for action_seqs in all_action_seqs:
            # print (start_state)
            action_seqs = list(action_seqs)

            ss1 = start_state[0]
            t1_s0_x,t1_s0_y = ss1
            action_seq_1 = action_seqs[0]
            traj1, t1_partial_r_sum, v_t1,is_terminal1,traj1_ts_x,traj1_ts_y, states1= create_traj(t1_s0_x, t1_s0_y,action_seq_1,traj_length, env.board, env.reward_function, env.terminal_cords,env.blocking_cords,V)
            if t1_partial_r_sum == False:
                # assert False
                # print ("here")
                continue

            ss2 = start_state[1]
            t2_s0_x,t2_s0_y = ss2
            action_seq_2 = action_seqs[1]
            traj2, t2_partial_r_sum, v_t2,is_terminal2,traj2_ts_x,traj2_ts_y, states2 = create_traj(t2_s0_x, t2_s0_y,action_seq_2,traj_length, env.board, env.reward_function, env.terminal_cords,env.blocking_cords,V)
            if t2_partial_r_sum == False:
                # assert False
                # print ("here")
                continue

            if traj1 == traj2:
                continue

            phi1,_ = find_reward_features(traj1, env, len(traj1)-1)
            phi2,_ = find_reward_features(traj2, env, len(traj2)-1)

            t1_partial_r_sum = np.dot(env.reward_array,phi1)
            t2_partial_r_sum = np.dot(env.reward_array,phi2)


            # only keep unique trajectorries
            big_traj_pair_tuple = ((traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y), (traj2[0][0],traj2[0][1]),(traj2_ts_x,traj2_ts_y), tuple(phi1), tuple(phi2))
            if big_traj_pair_tuple not in seen_traj_pairs:
                seen_traj_pairs.add(big_traj_pair_tuple)
            else:
                continue

            v_dif1 = v_t1 - V[t1_s0_x][t1_s0_y]
            v_dif2 = v_t2 - V[t2_s0_x][t2_s0_y]
            partial_sum_dif = float(t2_partial_r_sum - t1_partial_r_sum)
            v_dif = float(v_dif2 - v_dif1)

            traj1_ses = [(traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y)]
            traj2_ses = [(traj2[0][0],traj2[0][1]),(traj2_ts_x,traj2_ts_y)]

            quad = [traj1, traj2, v_dif, partial_sum_dif,(is_terminal1,is_terminal2)]

            all_collected_trajs.append(quad)

            all_X.append([phi1, phi2])
            all_r.append([t1_partial_r_sum, t2_partial_r_sum])
            all_ses.append([traj1_ses,traj2_ses])
            all_trajs.append([traj1,traj2])
            all_env_boards.append(env.board)


            n_pairs_found+=1
    return all_X, all_r, all_ses, all_trajs, all_env_boards


def generate_fig_8_MDP(rew_vec):
    #generate M_1 and M'_1 from figure 8
    width = 1
    height = 4
    env = GridWorldEnv(None, height, width)
    env.board[0][0] = 1
    env.board[3][0] = 3
    env.set_custom_reward_function(rew_vec,set_global=True)
    env.find_n_starts()
    env.get_blocking_cords()
    env.get_terminal_cords()
    all_X_, all_r_, all_ses_, all_trajs_,_ = get_env_trajs(env)

    all_trajs = []
    all_ses = []
    all_r = []
    all_X = []

    for i in range(len(all_trajs_)):
        if all_trajs_[i][0][0] == (2,0) and all_trajs_[i][1][0] == (2,0):
            all_trajs.append(all_trajs_[i])
            all_ses.append(all_ses_[i])
            all_r.append(all_r_[i])
            all_X.append(all_X_[i])
       

    print (len(all_X))
    return env, all_X, all_r, all_ses, all_trajs

# GAMMA = 0.999
# env1, all_X1, all_r1, all_ses1, all_trajs1  = generate_fig_8_MDP([-1,-1,-3,0,0,0])
# gt_rew_vec = env1.reward_array.copy()
# succ_feats, pis, succ_q_feats = generate_all_policies(25,GAMMA,env1,gt_rew_vec)
# np.save("random_MDPs/MDP_" + str(308) + "all_trajs.npy", all_trajs1)
# np.save("random_MDPs/MDP_" + str(308) + "succ_feats.npy", succ_feats)
# np.save("random_MDPs/MDP_" + str(308) + "succ_q_feats.npy", succ_q_feats)

# np.save("random_MDPs/MDP_" + str(308) + "gt_rew_vec.npy", gt_rew_vec)
# np.save("random_MDPs/MDP_" + str(308) + "all_X.npy", all_X1)
# np.save("random_MDPs/MDP_" + str(308) + "all_r.npy", all_r1)
# np.save("random_MDPs/MDP_" + str(308) + "all_ses.npy", all_ses1)
# with open(f'random_MDPs/MDP_' + str(308) + 'env.pickle', 'wb') as file:
#     pickle.dump(env1, file)

# env2, all_X2, all_r2, all_ses2, all_trajs2  = generate_fig_8_MDP([-1,-2,-3,0,0,0])
# gt_rew_vec = env2.reward_array.copy()
# succ_feats, pis, succ_q_feats = generate_all_policies(25,GAMMA,env2,gt_rew_vec)
# np.save("random_MDPs/MDP_" + str(309) + "all_trajs.npy", all_trajs2)
# np.save("random_MDPs/MDP_" + str(309) + "succ_feats.npy", succ_feats)
# np.save("random_MDPs/MDP_" + str(309) + "succ_q_feats.npy", succ_q_feats)

# np.save("random_MDPs/MDP_" + str(309) + "gt_rew_vec.npy", gt_rew_vec)
# np.save("random_MDPs/MDP_" + str(309) + "all_X.npy", all_X2)
# np.save("random_MDPs/MDP_" + str(309) + "all_r.npy", all_r2)
# np.save("random_MDPs/MDP_" + str(309) + "all_ses.npy", all_ses2)
# with open(f'random_MDPs/MDP_' + str(309) + 'env.pickle', 'wb') as file:
#     pickle.dump(env2, file)


for trial in range(0,30):
    print ("generating SF for MDP: " + str(trial))
    gt_rew_vec = np.load("random_MDPs/MDP_" + str(trial) +"gt_rew_vec.npy")
    with open("random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
        env = pickle.load(rf)
        env.generate_transition_probs()
        env.set_custom_reward_function(gt_rew_vec)
    
    for GAMMA in [0,0.5,0.9,0.99]:
        print ("    gamma=: " + str(GAMMA))
        succ_feats,sa_succ_feats, pis, succ_q_feats = generate_all_policies(100,GAMMA,env,gt_rew_vec)
        np.save("random_MDPs/MDP_" + str(trial) + "succ_feats_gamma=" + str(GAMMA)+ ".npy", succ_feats)
        np.save("random_MDPs/MDP_" + str(trial) + "succ_q_feats_gamma=" + str(GAMMA)+ ".npy", succ_q_feats)
        np.save("random_MDPs/MDP_" + str(trial) + "sa_succ_feats_gamma=" + str(GAMMA)+ ".npy", sa_succ_feats)


# GAMMA = 0.999
# for trial in range(0,30):
#     print ("generating MDP: " + str(trial))
#     env, all_X, all_r, all_ses,all_trajs = generate_MDP()
#     gt_rew_vec = env.reward_array.copy()

#     succ_feats,sa_succ_feats, pis, succ_q_feats = generate_all_policies(100,GAMMA,env,gt_rew_vec)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_trajs.npy", all_trajs)
#     # np.save("random_MDPs/MDP_" + str(trial) + "all_env_boards.npy", all_env_boards)
#     np.save("random_MDPs/MDP_" + str(trial) + "succ_feats.npy", succ_feats)
#     np.save("random_MDPs/MDP_" + str(trial) + "succ_q_feats.npy", succ_q_feats)
#     np.save("random_MDPs/MDP_" + str(trial) + "sa_succ_feats.npy", sa_succ_feats)

#     np.save("random_MDPs/MDP_" + str(trial) + "gt_rew_vec.npy", gt_rew_vec)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_X.npy", all_X)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_r.npy", all_r)
#     np.save("random_MDPs/MDP_" + str(trial) + "all_ses.npy", all_ses)
#     with open(f'random_MDPs/MDP_' + str(trial) + 'env.pickle', 'wb') as file:
#         pickle.dump(env, file)
