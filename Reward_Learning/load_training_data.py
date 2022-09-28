import numpy as np
import pickle
import re
import json
import math

with open('../MTURK_Data/2021_08_collected_data/2021_08_18_woi_questions.data', 'rb') as f:
    questions = pickle.load(f)
with open('../MTURK_Data/2021_08_collected_data/2021_08_18_woi_answers.data', 'rb') as f:
    answers = pickle.load(f)

with open('../MTURK_Data/2021_1_10_no_pen_0_questions.data', 'rb') as f:
    questions_aug = pickle.load(f)
with open('../MTURK_Data/2021_1_10_no_pen_0_answers.data', 'rb') as f:
    answers_aug = pickle.load(f)


dsdt_data = "../Segment_Selection/saved_data/2021_07_29_dsdt_chosen.json"
dsst_data = "../Segment_Selection/saved_data/2021_07_29_dsst_chosen.json"
ssst_data = "../Segment_Selection/saved_data/2021_07_29_ssst_chosen.json"
sss_data = "../Segment_Selection/saved_data/2021_07_29_sss_chosen.json"

multi_len_data = "../Segment_Selection/saved_data/augmented_trajs_multilength.json"
t_nt_1_data = "../Segment_Selection/saved_data/augmented_trajs_t_nt_quad1.json"
t_nt_2_data = "../Segment_Selection/saved_data/augmented_trajs_t_nt_quad2.json"


board = "../Segment_Selection/saved_data/2021-07-29_sparseboard2-notrap_board.json"
board_vf = "../Segment_Selection/saved_data/2021-07-29_sparseboard2-notrap_value_function.json"
board_rf = "../Segment_Selection/saved_data/2021-07-29_sparseboard2-notrap_rewards_function.json"



with open(board_vf, 'r') as j:
    board_vf = json.loads(j.read())

with open(dsdt_data, 'r') as j:
    dsdt_data = json.loads(j.read())
with open(dsst_data, 'r') as j:
    dsst_data = json.loads(j.read())
with open(ssst_data, 'r') as j:
    ssst_data = json.loads(j.read())
with open(sss_data, 'r') as j:
    sss_data = json.loads(j.read())

with open(multi_len_data, 'r') as j:
    multi_len_data = json.loads(j.read())
with open(t_nt_1_data, 'r') as j:
    t_nt_1_data = json.loads(j.read())
with open(t_nt_2_data, 'r') as j:
    t_nt_2_data = json.loads(j.read())

with open(board, 'r') as j:
    board = json.loads(j.read())
with open(board_rf, 'r') as j:
    board_rf = json.loads(j.read())


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

def is_in_blocked_area(x,y):
    val = board[x][y]
    if val == 2 or val == 8:
        return True
    else:
        return False

def get_state_feature(x,y,env=None):
    if env is not None:
        b = env.board
    else:
        b = board

    reward_feature = np.zeros(6)
    if b[x][y] == 0:
        reward_feature[0] = 1
    elif b[x][y] == 1:
        #flag
        # reward_feature[0] = 1
        reward_feature[1] = 1
    elif b[x][y] == 2:
        #house
        # reward_feature[0] = 1
        pass
    elif b[x][y] == 3:
        #sheep
        # reward_feature[0] = 1
        reward_feature[2] = 1
    elif b[x][y] == 4:
        #coin
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[3] = 1
    elif b[x][y] == 5:
        #road block
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[4] = 1
    elif b[x][y] == 6:
        #mud area
        # reward_feature[0] = 1
        reward_feature[5] = 1
    elif b[x][y] == 7:
        #mud area + flag
        reward_feature[1] = 1
    elif b[x][y] == 8:
        #mud area + house
        pass
    elif b[x][y] == 9:
        #mud area + sheep
        reward_feature[2] = 1
    elif b[x][y] == 10:
        #mud area + coin
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[3] = 1
    elif b[x][y] == 11:
        #mud area + roadblock
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[4] = 1
    return reward_feature

def get_action_feature(x,y,a,env=None):
    if env == None:
        arr = np.zeros((10,10,4))
    else:
        arr = np.zeros((env.height,env.width,4))
    arr[x][y][a] = 1
    return np.ravel(arr)

def get_traj_states(traj,traj_length=3):

    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    states = [[traj_ts_x, traj_ts_y]]

    prev_x = traj_ts_x
    prev_y = traj_ts_y
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    for i in range (1,traj_length+1):
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(board[0]) and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1]):
            if not(board[traj_ts_x][traj_ts_y] == 1 or board[traj_ts_x][traj_ts_y] == 3 or board[traj_ts_x][traj_ts_y] == 7 or board[traj_ts_x][traj_ts_y] == 9):
                traj_ts_x += traj[i][0]
                traj_ts_y += traj[i][1]
        states.append([traj_ts_x,traj_ts_y])
        prev_x = traj_ts_x
        prev_y = traj_ts_y
    return states



def find_reward_features(traj,traj_length=3,GAMMA=1,use_extended_SF=False):
    # GAMMA = 1 #overwrite GAMMA
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    if use_extended_SF:
        phi = np.zeros(6+400)
        phi_dis = np.zeros(6+400)
    else:
        phi = np.zeros(6)
        phi_dis = np.zeros(6)


    for i in range (1,traj_length+1):
        # print ("===========================")
        # print ("===========================")
        #check if we are at terminal state
        if board[traj_ts_x][traj_ts_y] == 1 or board[traj_ts_x][traj_ts_y] == 3 or board[traj_ts_x][traj_ts_y] == 7 or board[traj_ts_x][traj_ts_y] == 9:
            continue
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < len(board) and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < len(board[0]) and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1]):
            # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
            # if in_gated == False or  (in_gated and next_in_gated):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]
        if (traj_ts_x,traj_ts_y) != (prev_x,prev_y):
            # print ("state feature: ")
            # print (get_state_feature(traj_ts_x,traj_ts_y,env))
            dis_state_sf = (GAMMA**(i-1))*get_state_feature(traj_ts_x,traj_ts_y)
            state_sf = get_state_feature(traj_ts_x,traj_ts_y)
        else:
            #check if we are at terminal state
            if board[traj_ts_x][traj_ts_y] == 1 or board[traj_ts_x][traj_ts_y] == 3 or board[traj_ts_x][traj_ts_y] == 7 or board[traj_ts_x][traj_ts_y] == 9:
                # print ("in terminal")
                dis_state_sf = [0,0,0,0,0,0]
                state_sf = [0,0,0,0,0,0]
            else:
                dis_state_sf = (GAMMA**(i-1))*(get_state_feature(traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])
                state_sf = (get_state_feature(traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])

        if use_extended_SF:
            #find action index
            for a_i, action_ in enumerate(actions):
                if action_[0] == traj[i][0] and action_[1] == traj[i][1]:
                    action_index = a_i

            if board[prev_x][prev_y] == 1 or board[prev_x][prev_y] == 3 or board[prev_x][prev_y] == 7 or board[prev_x][prev_y] == 9:
                dis_action_sf = np.zeros(400)
            else:
                dis_action_sf =(GAMMA**(i-1))*get_action_feature(prev_x, prev_y, action_index)
            dis_state_sf = list(dis_state_sf)
            dis_state_sf.extend(dis_action_sf)

            if board[prev_x][prev_y] == 1 or board[prev_x][prev_y] == 3 or board[prev_x][prev_y] == 7 or board[prev_x][prev_y] == 9:
                action_sf = np.zeros(400)
            else:
                action_sf =get_action_feature(prev_x, prev_y, action_index)

            # print ("action_sf: " + str(action_sf))
            state_sf = list(state_sf)
            state_sf.extend(action_sf)

        phi_dis += dis_state_sf
        phi+= state_sf


        prev_x = traj_ts_x
        prev_y = traj_ts_y
    # print ("--done--\n")
    return phi_dis,phi


def find_end_state(traj,traj_length=3):
    in_gated =False
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y

    for i in range (1,traj_length+1):
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < 10 and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < 10 and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1]):
            # next_in_gated = is_in_gated_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1])
            # if in_gated == False or  (in_gated and next_in_gated):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]

            a = [traj_ts_x - prev_x, traj_ts_y - prev_y]
        else:
            a = [traj_ts_x + traj[i][0] - prev_x, traj_ts_y + traj[i][1] - prev_y]

        r = board_rf[prev_x][prev_y][find_action_index(a)]

        partial_return += r
        prev_x = traj_ts_x
        prev_y = traj_ts_y

    return traj_ts_x, traj_ts_y,partial_return



def generate_t_nt_samples(terminating, non_terminating):
    phis = []
    y = []
    rewards = []
    ses = []
    trajs = []
    actions = []
    states = []
    for i in range (min(len(terminating),len(non_terminating))):
        y.append(None)

        phis.append([terminating[i][0],non_terminating[i][0]])
        rewards.append([terminating[i][1], non_terminating[i][1]])
        ses.append([terminating[i][2],non_terminating[i][2]])
        trajs.append([terminating[i][3],non_terminating[i][3]])
        actions.append([terminating[i][4],non_terminating[i][4]])
        states.append([terminating[i][5],non_terminating[i][5]])

    return phis,y,rewards,ses,trajs,actions,states

def truncate_traj_1(traj,use_extended_SF,GAMMA):
    #truncate traj and get new start state
    x,y,_ = find_end_state (traj[:3],traj_length=2)
    traj= traj[2:]
    traj[0] = [x,y]

    traj_ts_x,traj_ts_y, pr =find_end_state(traj,traj_length=1)
    phi,_ = find_reward_features(traj,use_extended_SF=use_extended_SF,GAMMA=GAMMA,traj_length=1)
    return pr, phi,[(x,y),(traj_ts_x,traj_ts_y)],traj

def truncate_traj_2(traj,use_extended_SF,GAMMA):
    #truncate traj and get new start state
    x,y,_ = find_end_state (traj[:2],traj_length=1)
    traj= traj[1:]
    traj[0] = [x,y]

    traj_ts_x,traj_ts_y, pr =find_end_state(traj,traj_length=2)
    phi,_ = find_reward_features(traj,use_extended_SF=use_extended_SF,GAMMA=GAMMA,traj_length=2)
    return pr, phi,[(x,y),(traj_ts_x,traj_ts_y)],traj


def get_worker_pref_data(questions,answers,sample_folder,quad2data,include_dif_seg_lengths=True,GAMMA=0.999):
    '''
    Input:
        - questions: a list of the preference pairs shown to humans
        - answers: a list of human preferences
        - include_dif_seg_lengths: whether or not to include segments where one terminates earlier than the other
        - use_extended_SF: whether to use "extended successor features", where there is a feature for each possible (s,a) pair
        - GAMMA: discount factor

    Output:
        - For each condition shown to the user (no information shown, partial return info shown, value function info shown), we return the
            - X: list of features for each preference pair
            - r: list of partial return for each preference pair
            - ses: start and end state for each preference pair
            - as: actions taken for each preference pair
            - states: states vistied in each preference pair
    '''

    #choose whether to only include certain points or not
    points2picfrom = [(0,0),(0,-1),(0,-2),(0,-4),(1,-1),(2,-2),(3,-3),(4,-4),"positive_terminal_n_non_terminal"]
    pickpoints = False


    pr_X = []
    vf_X = []
    none_X = []

    pr_X_terminating = []
    vf_X_terminating = []
    none_X_terminating = []

    pr_X_non_terminating = []
    vf_X_non_terminating = []
    none_X_non_terminating = []

    pr_y = []
    vf_y = []
    none_y = []

    pr_r = []
    vf_r = []
    none_r = []

    pr_ses = []
    vf_ses = []
    none_ses = []

    pr_as = []
    vf_as = []
    none_as = []

    pr_states = []
    vf_states = []
    none_states = []

    n_incorrect = 0
    n_correct = 0
    total_ = 0
    for i in range(len(questions)):
        assignment_qs = questions[i]
        assignment_as = answers[i]
        sample_n = assignment_as[0]
        disp_id = None
        cords_id = [0,0]
        for q,a in zip(assignment_qs, assignment_as):
            if a == "dis":
                continue
            if q == "observationType":
                if a == "0":
                    disp_id = "pr"
                elif a == "1":
                    disp_id = "vf"
                elif a == "2":
                    disp_id = "none"
                else:
                    print (a)
                    print ("disp id error")
                # print (disp_id)
                cords_id[1] = int(a)
                continue
            if q == "sampleNumber":
                # print (a)
                cords_id[0] = int(a)
                continue

            sample_dict_path = "../MTURK_Data/" + sample_folder + "/"  + disp_id + "_sample" + str(sample_n) + "/" + "sample" + str(sample_n) + "_dict.pkl"

            with open(sample_dict_path, 'rb') as f:
                sample_dict = pickle.load(f)
            num = int(q.replace("query",""))
            point = sample_dict.get(num)
            quad = point.get("quadrant")


            split_name = point.get("name").split("/")[-1].split("_")
            if (split_name[0] == "vf" or split_name[0] == "none"):
                pt = split_name[1]
                index = split_name[2]
            else:
                pt = split_name[0]
                index = split_name[1]

            quad = quad.replace("_formatted_imgs","") #bug from some augmented human data formatting

            if quad not in quad2data:
                print ("quad not present")
                continue
            traj_pairs = quad2data[quad].get(pt)


            pt_ = pt.replace("(","")
            pt_ = pt_.replace(")","")
            pt_ = pt_.split(",")
            x = float(pt_[0])
            y = float(pt_[1])
            

            if pickpoints:
                if (x,y) not in points2picfrom and ("positive_terminal_n_non_terminal" in points2picfrom and not(y < 0 and x >30)):
                    continue


            poi = traj_pairs[int(index)]
            traj1 = poi[0]
            traj1_ts_x,traj1_ts_y, pr1 =find_end_state(traj1,len(traj1)-1)
            traj1_v_s0 = board_vf[traj1[0][0]][traj1[0][1]]
            traj1_v_st = board_vf[traj1_ts_x][traj1_ts_y]
            phi1,phi1_nd = find_reward_features(traj1,len(traj1)-1, GAMMA=GAMMA)

            traj2 = poi[1]
            traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2,len(traj2)-1)
            traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
            traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]
            phi2,phi2_nd = find_reward_features(traj2,len(traj2)-1, GAMMA=GAMMA)

            disp_type = point.get("disp_type")
            dom_val = point.get("dom_val")

            if dom_val == "R":
                if a == "left":
                    a = "right"
                elif a == "right":
                    a = "left"

            #make sure that our calculated pr/sv are the same as what the trajectory pair is marked as
            #sometimes things get flipped (bug in how augmented data was formatted after flipping), so flip back
            if not ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x):
                traj1 = poi[1]
                traj1_ts_x,traj1_ts_y, pr1 =find_end_state(traj1,len(traj1)-1)
                traj1_v_s0 = board_vf[traj1[0][0]][traj1[0][1]]
                traj1_v_st = board_vf[traj1_ts_x][traj1_ts_y]
                phi1 = find_reward_features(traj1,len(traj1)-1)
                phi1,phi1_nd = find_reward_features(traj1,len(traj1)-1, GAMMA=GAMMA)

                traj2 = poi[0]
                traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2,len(traj2)-1)
                traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
                traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]
                phi2 = find_reward_features(traj2,len(traj2)-1)
                phi2,phi2_nd = find_reward_features(traj2,len(traj2)-1, GAMMA=GAMMA)

                if a == "left":
                    a = "right"
                elif a == "right":
                    a = "left"


            assert ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x)
            assert (pr2 - pr1 == y)

            if (pr1 != np.dot(phi1_nd, [-1,50,-50,1,-1,-2])) or (pr2 != np.dot(phi2_nd, [-1,50,-50,1,-1,-2])):
                n_incorrect+=1
                continue
            
            if not include_dif_seg_lengths:
                if abs(pr1-pr2) >= 30:
                    continue


            if a == "left":
                encoded_a = 0
                # encoded_a = [1,0]
            elif a == "right":
                encoded_a = 1
                # encoded_a = [0,1]
            elif a == "same":
                encoded_a = 0.5
                # encoded_a = [0.5,0.5]
            else:
                # print (a)
                encoded_a = None
            traj1_ses = [(traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y)]
            traj2_ses = [(traj2[0][0],traj2[0][1]),(traj2_ts_x,traj2_ts_y)]

            if disp_id == "vf":
                vf_X.append([phi1,phi2])
                vf_r.append([pr1,pr2])
                vf_y.append(encoded_a)
                vf_ses.append([traj1_ses,traj2_ses])
                vf_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2[1:])])
                vf_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2,len(traj2)-1)])

            elif disp_id == "pr":
                pr_X.append([phi1,phi2])
                pr_r.append([pr1,pr2])
                pr_y.append(encoded_a)
                pr_ses.append([traj1_ses,traj2_ses])
                pr_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2[1:])])
                pr_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2,len(traj2)-1)])

            elif disp_id == "none":
                none_X.append([phi1,phi2])
                none_r.append([pr1,pr2])
                none_y.append(encoded_a)
                none_ses.append([traj1_ses,traj2_ses])
                none_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2[1:])])
                none_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2,len(traj2)-1)])

                

    return vf_X, vf_r, vf_y, vf_ses, vf_as, vf_states, pr_X, pr_r, pr_y, pr_ses, pr_as, pr_states, none_X, none_r, none_y, none_ses, none_as, none_states


def get_all_statistics_aug_human(include_dif_seg_lengths=True,GAMMA=0.999):
    '''
    Collects all human preference data from both stages of data collection
    '''
    quad2data1 = {"dsdt":dsdt_data,"dsst":dsst_data,"ssst":ssst_data, "sss":sss_data}
    sample_folder1 = "2021_07_29_data_samples"


    quad2data2 = {"aug-mul-len":multi_len_data, "aug-t-nt-1":t_nt_1_data, "aug-t-nt-2":t_nt_2_data}
    sample_folder2 = "2021_12_22_data_samples"
    

    vf_X, vf_r, vf_y, vf_ses, vf_as, vf_states, pr_X, pr_r, pr_y, pr_ses, pr_as, pr_states, none_X, none_r, none_y, none_ses, none_as, none_states = get_worker_pref_data(questions,answers,sample_folder1,quad2data1,include_dif_seg_lengths,GAMMA)
    print("Non augmenting prefs:", len(none_X))
    vf_X2, vf_r2, vf_y2, vf_ses2, vf_as2, vf_states2, pr_X2, pr_r2, pr_y2, pr_ses2, pr_as2, pr_states2, none_X2, none_r2, none_y2, none_ses2, none_as2, none_states2 = get_worker_pref_data(questions_aug,answers_aug,sample_folder2,quad2data2,include_dif_seg_lengths,GAMMA)

    # include augmenting data
    vf_X.extend(vf_X2)
    vf_r.extend(vf_r2)
    vf_y.extend(vf_y2)
    vf_ses.extend(vf_ses2)
    vf_as.extend(vf_as2)
    vf_states.extend(vf_states2)
    pr_X.extend(pr_X2)
    pr_r.extend(pr_r2)
    pr_y.extend(pr_y2)
    pr_ses.extend(pr_ses2)
    pr_as.extend(pr_as2)
    pr_states.extend(pr_states2)
    none_X.extend(none_X2)
    none_r.extend(none_r2)
    none_y.extend(none_y2)
    none_ses.extend(none_ses2)
    none_as.extend(none_as2)
    none_states.extend(none_states2)
    return vf_X, vf_r, vf_y, vf_ses, vf_as, vf_states, pr_X, pr_r, pr_y, pr_ses, pr_as, pr_states, none_X, none_r, none_y, none_ses,none_as, none_states


def get_all_statistics(questions=questions,answers=answers,include_dif_seg_lengths = False,use_extended_SF=False,GAMMA=1):
    '''
    Input:
        - questions: a list of the preference pairs shown to humans
        - answers: a list of human preferences
        - include_dif_seg_lengths: whether or not to include segments where one terminates earlier than the other
        - use_extended_SF: whether to use "extended successor features", where there is a feature for each possible (s,a) pair
        - GAMMA: discount factor

    Output:
        - For each condition shown to the user (no information shown, partial return info shown, value function info shown), we return the
            - X: list of features for each preference pair
            - r: list of partial return for each preference pair
            - ses: start and end state for each preference pair
            - as: actions taken for each preference pair
            - states: states vistied in each preference pair
    '''
    pr_X = []
    vf_X = []
    none_X = []


    pr_X_terminating = []
    vf_X_terminating = []
    none_X_terminating = []

    pr_X_non_terminating = []
    vf_X_non_terminating = []
    none_X_non_terminating = []

    pr_y = []
    vf_y = []
    none_y = []

    pr_r = []
    vf_r = []
    none_r = []

    pr_ses = []
    vf_ses = []
    none_ses = []


    pr_as = []
    vf_as = []
    none_as = []

    pr_states = []
    vf_states = []
    none_states = []

    augmented_segs = []

    n_incorrect = 0
    for i in range(len(questions)):
        assignment_qs = questions[i]
        assignment_as = answers[i]
        sample_n = assignment_as[0]
        disp_id = None
        cords_id = [0,0]
        for q,a in zip(assignment_qs, assignment_as):
            if a == "dis":
                continue
            if q == "observationType":
                if a == "0":
                    disp_id = "pr"
                elif a == "1":
                    disp_id = "vf"
                elif a == "2":
                    disp_id = "none"
                else:
                    print (a)
                    print ("disp id error")
                # print (disp_id)
                cords_id[1] = int(a)
                continue
            if q == "sampleNumber":
                # print (a)
                cords_id[0] = int(a)
                continue

            sample_dict_path = "../MTURK_Data/2021_07_29_data_samples/"  + disp_id + "_sample" + str(sample_n) + "/" + "sample" + str(sample_n) + "_dict.pkl"

            with open(sample_dict_path, 'rb') as f:
                sample_dict = pickle.load(f)
            num = int(q.replace("query",""))
            point = sample_dict.get(num)
            quad = point.get("quadrant")

            split_name = point.get("name").split("/")[-1].split("_")
            if (split_name[0] == "vf" or split_name[0] == "none"):
                pt = split_name[1]
                index = split_name[2]
            else:
                pt = split_name[0]
                index = split_name[1]


            if quad == "dsdt":
                traj_pairs = dsdt_data.get(pt)
            if quad == "dsst":
                traj_pairs = dsst_data.get(pt)
            if quad == "ssst":
                traj_pairs = ssst_data.get(pt)
            if quad == "sss":
                traj_pairs = sss_data.get(pt)


            pt_ = pt.replace("(","")
            pt_ = pt_.replace(")","")
            pt_ = pt_.split(",")
            x = float(pt_[0])
            y = float(pt_[1])

            poi = traj_pairs[int(index)]
            traj1 = poi[0]
            traj1_ts_x,traj1_ts_y, pr1 =find_end_state(traj1)
            traj1_v_s0 = board_vf[traj1[0][0]][traj1[0][1]]
            traj1_v_st = board_vf[traj1_ts_x][traj1_ts_y]

            phi1,phi1_nd = find_reward_features(traj1,use_extended_SF=use_extended_SF,GAMMA=GAMMA)

            traj2 = poi[1]
            traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2)
            traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
            traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]

            phi2,ph2_nd = find_reward_features(traj2,use_extended_SF=use_extended_SF,GAMMA=GAMMA)

            #make sure that our calculated pr/sv are the same as what the trajectory pair is marked as
            assert ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x)
            assert (pr2 - pr1 == y)

            #get truncated trajectories
            os_trunc_traj1_pr,os_trunc_traj1_phi, os_ses1, traj1_1_trunc = truncate_traj_1(traj1,use_extended_SF,GAMMA)
            os_trunc_traj2_pr,os_trunc_traj2_phi, os_ses2, traj1_2_trunc = truncate_traj_1(traj2,use_extended_SF,GAMMA)

            ts_trunc_traj1_pr,ts_trunc_traj1_phi, ts_ses1, traj2_1_trunc = truncate_traj_2(traj1,use_extended_SF,GAMMA)
            ts_trunc_traj2_pr,ts_trunc_traj2_phi, ts_ses2, traj2_2_trunc = truncate_traj_2(traj2,use_extended_SF,GAMMA)

            #Remove corrupted segment pairs (somehow somewhere something went wrong with the stored meta-data for a handful of segment pairs)
            if not use_extended_SF:
                if (pr1 != np.dot(phi1_nd, [-1,50,-50,1,-1,-2])) or (pr2 != np.dot(ph2_nd, [-1,50,-50,1,-1,-2])):
                    n_incorrect+=1
                    continue
            if use_extended_SF:
                if(pr1 != np.dot(phi1_nd[0:6], [-1,50,-50,1,-1,-2])) or (pr2 != np.dot(ph2_nd[0:6], [-1,50,-50,1,-1,-2])):
                    n_incorrect+=1
                    continue

            # #change partial return to discounted partial return
            # pr1 = np.dot(phi1[0:6], [-1,50,-50,1,-1,-2])
            # pr2 = np.dot(phi2[0:6], [-1,50,-50,1,-1,-2])

            disp_type = point.get("disp_type")
            dom_val = point.get("dom_val")

            if dom_val == "R":
                if a == "left":
                    a = "right"
                elif a == "right":
                    a = "left"

            if a == "left":
                encoded_a = 0
                # encoded_a = [1,0]
            elif a == "right":
                encoded_a = 1
                # encoded_a = [0,1]
            elif a == "same":
                encoded_a = 0.5
                # encoded_a = [0.5,0.5]
            else:
                # print (a)
                encoded_a = None
            traj1_ses = [(traj1[0][0],traj1[0][1]),(traj1_ts_x,traj1_ts_y)]
            traj2_ses = [(traj2[0][0],traj2[0][1]),(traj2_ts_x,traj2_ts_y)]

            if disp_id == "vf":
                vf_X.append([phi1,phi2])
                vf_r.append([pr1,pr2])
                vf_y.append(encoded_a)
                vf_ses.append([traj1_ses,traj2_ses])
                vf_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2[1:])])
                vf_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2,len(traj2)-1)])
                if quad == "dsst" or quad == "ssst":
                    vf_X_terminating.append([phi1,pr1,traj1_ses,traj1,get_action_indices(traj1[1:]), get_traj_states(traj1,len(traj1)-1)])
                    vf_X_terminating.append([phi2,pr2,traj2_ses,traj2,get_action_indices(traj2[1:]), get_traj_states(traj2,len(traj2)-1)])

                    if include_dif_seg_lengths:
                        vf_X.append([phi1, os_trunc_traj1_phi])
                        vf_r.append([pr1, os_trunc_traj1_pr])
                        vf_y.append(None)
                        vf_ses.append([traj1_ses,os_ses1])
                        vf_as.append([get_action_indices(traj1[1:]),get_action_indices(traj1_1_trunc[1:])])
                        vf_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj1_1_trunc,len(traj1_1_trunc)-1)])
                        # augmented_segs.append([traj1, traj1_1_trunc])


                        vf_X.append([phi1, ts_trunc_traj1_phi])
                        vf_r.append([pr1, ts_trunc_traj1_pr])
                        vf_y.append(None)
                        vf_ses.append([traj1_ses,ts_ses1])
                        vf_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2_1_trunc[1:])])
                        vf_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2_1_trunc,len(traj2_1_trunc)-1)])
                        # augmented_segs.append([traj1, traj1_2_trunc])


                        vf_X.append([phi2, os_trunc_traj2_phi])
                        vf_r.append([pr2, os_trunc_traj2_pr])
                        vf_y.append(None)
                        vf_ses.append([traj2_ses,os_ses2])
                        vf_as.append([get_action_indices(traj2[1:]),get_action_indices(traj1_2_trunc[1:])])
                        # augmented_segs.append([traj2, traj2_1_trunc])
                        vf_states.append([get_traj_states(traj2,len(traj2)-1), get_traj_states(traj1_2_trunc,len(traj1_2_trunc)-1)])

                        vf_X.append([phi2, ts_trunc_traj2_phi])
                        vf_r.append([pr2, ts_trunc_traj2_pr])
                        vf_y.append(None)
                        vf_ses.append([traj2_ses,ts_ses2])
                        vf_as.append([get_action_indices(traj2[1:]),get_action_indices(traj2_2_trunc[1:])])
                        vf_states.append([get_traj_states(traj2,len(traj2)-1), get_traj_states(traj2_2_trunc,len(traj2_2_trunc)-1)])

                        # augmented_segs.append([traj2, traj2_2_trunc])

                elif quad == "dsdt" or quad == "sss":
                    vf_X_non_terminating.append([phi1,pr1,traj1_ses,traj1,get_action_indices(traj1[1:]), get_traj_states(traj1,len(traj1)-1)])
                    vf_X_non_terminating.append([phi2,pr2,traj2_ses,traj2,get_action_indices(traj2[1:]), get_traj_states(traj2,len(traj2)-1)])

            elif disp_id == "pr":
                pr_X.append([phi1,phi2])
                pr_r.append([pr1,pr2])
                pr_y.append(encoded_a)
                pr_ses.append([traj1_ses,traj2_ses])
                pr_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2[1:])])
                pr_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2,len(traj2)-1)])

                if quad == "dsst" or quad == "ssst":
                    pr_X_terminating.append([phi1, pr1, traj1_ses, traj1,get_action_indices(traj1[1:]), get_traj_states(traj1,len(traj1)-1)])
                    pr_X_terminating.append([phi2, pr2, traj2_ses, traj2,get_action_indices(traj2[1:]), get_traj_states(traj2,len(traj2)-1)])

                    if include_dif_seg_lengths:
                        pr_X.append([phi1, os_trunc_traj1_phi])
                        pr_r.append([pr1, os_trunc_traj1_pr])
                        pr_y.append(None)
                        pr_ses.append([traj1_ses, os_ses1])
                        pr_as.append([get_action_indices(traj1[1:]),get_action_indices(traj1_1_trunc[1:])])
                        pr_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj1_1_trunc,len(traj1_1_trunc)-1)])

                        # augmented_segs.append([traj1, traj1_1_trunc])

                        pr_X.append([phi1, ts_trunc_traj1_phi])
                        pr_r.append([pr1, ts_trunc_traj1_pr])
                        pr_y.append(None)
                        pr_ses.append([traj1_ses, ts_ses1])
                        # augmented_segs.append([traj1, traj1_2_trunc])
                        pr_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2_1_trunc[1:])])
                        pr_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2_1_trunc,len(traj2_1_trunc)-1)])


                        pr_X.append([phi2, os_trunc_traj2_phi])
                        pr_r.append([pr2, os_trunc_traj2_pr])
                        pr_y.append(None)
                        pr_ses.append([traj2_ses, os_ses2])
                        # augmented_segs.append([traj2, traj2_1_trunc])
                        pr_as.append([get_action_indices(traj2[1:]),get_action_indices(traj1_2_trunc[1:])])
                        pr_states.append([get_traj_states(traj2,len(traj2)-1), get_traj_states(traj1_2_trunc,len(traj1_2_trunc)-1)])


                        pr_X.append([phi2, ts_trunc_traj2_phi])
                        pr_r.append([pr2, ts_trunc_traj2_pr])
                        pr_y.append(None)
                        pr_ses.append([traj2_ses, ts_ses2])
                        # augmented_segs.append([traj2, traj2_2_trunc])
                        pr_as.append([get_action_indices(traj2[1:]),get_action_indices(traj2_2_trunc[1:])])
                        pr_states.append([get_traj_states(traj2,len(traj2)-1), get_traj_states(traj2_2_trunc,len(traj2_2_trunc)-1)])


                elif quad == "dsdt" or quad == "sss":
                    pr_X_non_terminating.append([phi1 ,pr1, traj1_ses, traj1,get_action_indices(traj1[1:]), get_traj_states(traj1,len(traj1)-1)])
                    pr_X_non_terminating.append([phi2, pr2, traj2_ses, traj2,get_action_indices(traj2[1:]), get_traj_states(traj2,len(traj2)-1)])

            elif disp_id == "none":
                none_X.append([phi1,phi2])
                none_r.append([pr1,pr2])
                none_y.append(encoded_a)
                none_ses.append([traj1_ses,traj2_ses])
                none_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2[1:])])
                none_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2,len(traj2)-1)])

                if quad == "dsst" or quad == "ssst":
                    none_X_terminating.append([phi1, pr1, traj1_ses, traj1,get_action_indices(traj1[1:]), get_traj_states(traj1,len(traj1)-1)])
                    none_X_terminating.append([phi2, pr2, traj2_ses, traj2,get_action_indices(traj2[1:]), get_traj_states(traj2,len(traj2)-1)])

                    if include_dif_seg_lengths:
                        none_X.append([phi1, os_trunc_traj1_phi])
                        none_r.append([pr1, os_trunc_traj1_pr])
                        none_y.append(None)
                        none_ses.append([traj1_ses, os_ses1])
                        augmented_segs.append([traj1, traj1_1_trunc])
                        none_as.append([get_action_indices(traj1[1:]),get_action_indices(traj1_1_trunc[1:])])
                        none_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj1_1_trunc,len(traj1_1_trunc)-1)])

                        none_X.append([phi1, ts_trunc_traj1_phi])
                        none_r.append([pr1, ts_trunc_traj1_pr])
                        none_y.append(None)
                        none_ses.append([traj1_ses, ts_ses1])
                        augmented_segs.append([traj1, traj1_2_trunc])
                        none_as.append([get_action_indices(traj1[1:]),get_action_indices(traj2_1_trunc[1:])])
                        none_states.append([get_traj_states(traj1,len(traj1)-1), get_traj_states(traj2_1_trunc,len(traj2_1_trunc)-1)])


                        none_X.append([phi2, os_trunc_traj2_phi])
                        none_r.append([pr2, os_trunc_traj2_pr])
                        none_y.append(None)
                        none_ses.append([traj2_ses, os_ses2])
                        augmented_segs.append([traj2, traj2_1_trunc])
                        none_as.append([get_action_indices(traj2[1:]),get_action_indices(traj1_2_trunc[1:])])
                        none_states.append([get_traj_states(traj2,len(traj2)-1), get_traj_states(traj1_2_trunc,len(traj1_2_trunc)-1)])


                        none_X.append([phi2, ts_trunc_traj2_phi])
                        none_r.append([pr2, ts_trunc_traj2_pr])
                        none_y.append(None)
                        none_ses.append([traj2_ses, ts_ses2])
                        augmented_segs.append([traj2, traj2_2_trunc])
                        none_as.append([get_action_indices(traj2[1:]),get_action_indices(traj2_2_trunc[1:])])
                        none_states.append([get_traj_states(traj2,len(traj2)-1), get_traj_states(traj2_2_trunc,len(traj2_2_trunc)-1)])

                elif quad == "dsdt" or quad == "sss":
                    none_X_non_terminating.append([phi1, pr1, traj1_ses, traj1,get_action_indices(traj1[1:]), get_traj_states(traj1,len(traj1)-1)])
                    none_X_non_terminating.append([phi2, pr2, traj2_ses, traj2,get_action_indices(traj2[1:]), get_traj_states(traj2,len(traj2)-1)])

    vf_X_add, vf_y_add, vf_add_r, vf_add_ses, vf_trajs, vf_add_as, vf_add_states = generate_t_nt_samples(vf_X_terminating, vf_X_non_terminating)
    pr_X_add, pr_y_add, pr_add_r, pr_add_ses, pr_trajs, pr_add_as, pr_add_states = generate_t_nt_samples(pr_X_terminating, pr_X_non_terminating)
    none_X_add, none_y_add, none_add_r, none_add_ses,none_trajs, none_add_as, none_add_states = generate_t_nt_samples(none_X_terminating, none_X_non_terminating)

    #find unique trajs in (vf_trajs, pr_trajs, none_trajs)
    t_nt_trajs = []
    seen_trajs = []
    # for t_ in vf_trajs:
    #     if (str(t_) in seen_trajs):
    #         continue
    #     else:
    #         seen_trajs.append(str(t_))
    #         t_nt_trajs.append(t_)
    # for t_ in pr_trajs:
    #     if (str(t_) in seen_trajs):
    #         continue
    #     else:
    #         seen_trajs.append(str(t_))
    #         t_nt_trajs.append(t_)
    # for t_ in none_trajs:
    #     if (str(t_) in seen_trajs):
    #         continue
    #     else:
    #         seen_trajs.append(str(t_))
    #         t_nt_trajs.append(t_)

    # augmented_segs.extend(t_nt_trajs)

    # adds syntheitc preferences between termianting and non-terminating trajectory
    vf_X.extend(vf_X_add)
    vf_r.extend(vf_add_r)
    vf_y.extend(vf_y_add)
    vf_ses.extend(vf_add_ses)
    vf_as.extend(vf_add_as)
    vf_states.extend(vf_add_states)

    pr_X.extend(pr_X_add)
    pr_r.extend(pr_add_r)
    pr_y.extend(pr_y_add)
    pr_ses.extend(pr_add_ses)
    pr_as.extend(pr_add_as)
    pr_states.extend(pr_add_states)

    none_X.extend(none_X_add)
    none_r.extend(none_add_r)
    none_y.extend(none_y_add)
    none_ses.extend(none_add_ses)
    none_as.extend(none_add_as)
    none_states.extend(none_add_states)

    np.save("augmented_traj_pairs_none.npy", augmented_segs)

    return vf_X, vf_r, vf_y, vf_ses,vf_as, vf_states, pr_X, pr_r, pr_y, pr_ses,pr_as,pr_states, none_X, none_r, none_y, none_ses,none_as,none_states
