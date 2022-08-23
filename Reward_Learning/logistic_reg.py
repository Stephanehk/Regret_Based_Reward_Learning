import pickle
import re
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
from scipy import stats
from mord import LogisticAT
from sklearn import metrics
from sklearn.model_selection import train_test_split
import json
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn import preprocessing
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import random
import math

from utils import format_X_pr, format_X_regret, format_X_full, sigmoid

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

def add2dict(pt,a,dict):
    if pt not in dict:
        dict[pt] = [a]
    else:
        arr = dict.get(pt)
        arr.append(a)
        dict[pt] = arr
    return dict

def find_action_index(action):
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    i = 0
    for a in actions:
        if a[0] == action[0] and a[1] == action[1]:
            return i
        i+=1
    return False

def is_in_blocked_area(x,y):
    val = board[x][y]
    if val == 2 or val == 8:
        return True
    else:
        return False
def get_state_feature(x,y):
    reward_feature = np.zeros(6)
    if board[x][y] == 0:
        reward_feature[0] = 1
    elif board[x][y] == 1:
        #flag
        # reward_feature[0] = 1
        reward_feature[1] = 1
    elif board[x][y] == 2:
        #house
        # reward_feature[0] = 1
        pass
    elif board[x][y] == 3:
        #sheep
        # reward_feature[0] = 1
        reward_feature[2] = 1
    elif board[x][y] == 4:
        #coin
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[3] = 1
    elif board[x][y] == 5:
        #road block
        # reward_feature[0] = 1
        reward_feature[0] = 1
        reward_feature[4] = 1
    elif board[x][y] == 6:
        #mud area
        # reward_feature[0] = 1
        reward_feature[5] = 1
    elif board[x][y] == 7:
        #mud area + flag
        reward_feature[1] = 1
    elif board[x][y] == 8:
        #mud area + house
        pass
    elif board[x][y] == 9:
        #mud area + sheep
        reward_feature[2] = 1
    elif board[x][y] == 10:
        #mud area + coin
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[3] = 1
    elif board[x][y] == 11:
        #mud area + roadblock
        # reward_feature[0] = 1
        reward_feature[5] = 1
        reward_feature[4] = 1
    return reward_feature


def find_reward_features(traj,traj_length=3):
    traj_ts_x = traj[0][0]
    traj_ts_y = traj[0][1]
    # if is_in_gated_area(traj_ts_x,traj_ts_y):
    #     in_gated = True

    partial_return = 0
    prev_x = traj_ts_x
    prev_y = traj_ts_y

    phi = np.zeros(6)

    for i in range (1,traj_length+1):
        if traj_ts_x + traj[i][0] >= 0 and traj_ts_x + traj[i][0] < 10 and traj_ts_y + traj[i][1] >=0 and traj_ts_y + traj[i][1] < 10 and not is_in_blocked_area(traj_ts_x + traj[i][0], traj_ts_y + traj[i][1]):
            traj_ts_x += traj[i][0]
            traj_ts_y += traj[i][1]
        if (traj_ts_x,traj_ts_y) != (prev_x,prev_y):
            phi += get_state_feature(traj_ts_x,traj_ts_y)
        else:
            #only keep the gas/mud area score
            phi += (get_state_feature(traj_ts_x,traj_ts_y)*[1,0,0,0,0,1])

        prev_x = traj_ts_x
        prev_y = traj_ts_y

    return phi

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

def get_all_statistics(questions,answers):
    pr_dsdt_res = {}
    pr_dsst_res = {}
    pr_ssst_res = {}
    pr_sss_res = {}

    vf_dsdt_res = {}
    vf_dsst_res = {}
    vf_ssst_res = {}
    vf_sss_res = {}

    none_dsdt_res = {}
    none_dsst_res = {}
    none_ssst_res = {}
    none_sss_res = {}

    prefrences = []
    delta_rs = []
    delta_v_sts = []
    delta_v_s0s = []
    delta_cis = []

    prefrences_dict = {}
    delta_rs_dict = {}
    delta_v_sts_dict = {}
    delta_v_s0s_dict = {}
    delta_cis_dict = {}


    for i in range(len(questions)):
        assignment_qs = questions[i]
        assignment_as = answers[i]
        sample_n = assignment_as[0]
        disp_id = None
        cords_id = [0,0]
        for q,a in zip(assignment_qs, assignment_as):
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

            sample_dict_path = "../2021_07_29_data_samples/"  + disp_id + "_sample" + str(sample_n) + "/" + "sample" + str(sample_n) + "_dict.pkl"

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

            traj2 = poi[1]
            traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2)
            traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
            traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]

            #make sure that our calculated pr/sv are the same as what the trajectory pair is marked as
            assert ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x)
            assert (pr2 - pr1 == y)

            disp_type = point.get("disp_type")
            dom_val = point.get("dom_val")

            delta_r = pr2 - pr1
            delta_v_s0 = traj2_v_s0 - traj1_v_s0
            delta_v_st = traj2_v_st - traj1_v_st

            if dom_val == "R":
                if a == "left":
                    a = "right"
                elif a == "right":
                    a = "left"

            if a == "left":
                encoded_a = 0
            elif a == "right":
                encoded_a = 1
            elif a == "same":
                encoded_a = 0.5
            else:
                encoded_a = None

            if encoded_a != None:
                prefrences_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, encoded_a, prefrences_dict)
                delta_rs_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, delta_r, delta_rs_dict)
                delta_v_sts_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, delta_v_st, delta_v_sts_dict)
                delta_v_s0s_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, delta_v_s0, delta_v_s0s_dict)
                delta_cis_dict = add2dict(str(disp_type) + "_" + quad + "_" + pt, x, delta_cis_dict)
                prefrences.append(encoded_a)
                delta_rs.append(delta_r)
                delta_v_sts.append(delta_v_st)
                delta_v_s0s.append(delta_v_s0)
                delta_cis.append(x)


                if quad == "dsdt":
                    if disp_type == 0:
                        pr_dsdt_res = add2dict(pt,encoded_a,pr_dsdt_res)
                    elif disp_type == 1:
                        vf_dsdt_res = add2dict(pt,encoded_a,vf_dsdt_res)
                    elif disp_type == 3:
                        none_dsdt_res = add2dict(pt,encoded_a,none_dsdt_res)
                    else:
                        print ("KEY ERROR")

                elif quad == "dsst":
                    if disp_type == 0:
                        pr_dsst_res = add2dict(pt,encoded_a,pr_dsst_res)
                    elif disp_type == 1:
                        vf_dsst_res = add2dict(pt,encoded_a,vf_dsst_res)
                    elif disp_type == 3:
                        none_dsst_res = add2dict(pt,encoded_a,none_dsst_res)
                    else:
                        print ("KEY ERROR")
                elif quad == "ssst":
                    if disp_type == 0:
                        pr_ssst_res = add2dict(pt,encoded_a,pr_ssst_res)
                    elif disp_type == 1:
                        vf_ssst_res = add2dict(pt,encoded_a,vf_ssst_res)
                    elif disp_type == 3:
                        none_ssst_res = add2dict(pt,encoded_a,none_ssst_res)
                    else:
                        print ("KEY ERROR")
                elif quad == "sss":
                    if disp_type == 0:
                        pr_sss_res = add2dict(pt,encoded_a,pr_sss_res)
                    elif disp_type == 1:
                        vf_sss_res = add2dict(pt,encoded_a,vf_sss_res)
                    elif disp_type == 3:
                        none_sss_res = add2dict(pt,encoded_a,none_sss_res)
                    else:
                        print ("KEY ERROR")


    return [delta_rs_dict, delta_v_sts_dict, delta_v_s0s_dict],delta_cis_dict, prefrences_dict, prefrences, delta_rs, delta_v_sts, delta_v_s0s, delta_cis,pr_dsdt_res,pr_dsst_res,pr_ssst_res,pr_sss_res, vf_dsdt_res,vf_dsst_res,vf_ssst_res,vf_sss_res, none_dsdt_res,none_dsst_res,none_ssst_res,none_sss_res


def get_worker_pref_data(questions,answers,sample_folder,quad2data):
    pr_r = []
    pr_vs0 = []
    pr_vst = []
    pr_pref = []
    pr_res = {}

    vf_r = []
    vf_vs0 = []
    vf_vst = []
    vf_pref = []
    vf_res = {}

    none_r = []
    none_vs0 = []
    none_vst = []
    none_pref = []
    none_res = {}

    n_incorrect = 0
    n_correct = 0
    total_ = 0
    incorrect = 0
    total = 0

    n_lefts_pref = 0
    n_rights_pref = 0
    n_none_users = 0

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
                    n_none_users+=1
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
            total_+=1


            split_name = point.get("name").split("/")[-1].split("_")
            if (split_name[0] == "vf" or split_name[0] == "none"):
                pt = split_name[1]
                index = split_name[2]
            else:
                pt = split_name[0]
                index = split_name[1]

            quad = quad.replace("_formatted_imgs","") #bug from some augmented human data formatting
            traj_pairs = quad2data[quad].get(pt)

            # if quad == "aug-mul-len" or quad == "aug-t-nt-1" or quad == "aug-t-nt-2":

            # if quad == "aug-mul-len":
            #     continue

            #problem is with aug-t-nt-1

            pt_ = pt.replace("(","")
            pt_ = pt_.replace(")","")
            pt_ = pt_.split(",")
            x = float(pt_[0])
            y = float(pt_[1])



            poi = traj_pairs[int(index)]
            traj1 = poi[0]
            traj1_ts_x,traj1_ts_y, pr1 =find_end_state(traj1,len(traj1)-1)
            traj1_v_s0 = board_vf[traj1[0][0]][traj1[0][1]]
            traj1_v_st = board_vf[traj1_ts_x][traj1_ts_y]
            phi1 = find_reward_features(traj1,len(traj1)-1)

            traj2 = poi[1]
            traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2,len(traj2)-1)
            traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
            traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]
            phi2 = find_reward_features(traj2,len(traj2)-1)

            disp_type = point.get("disp_type")
            dom_val = point.get("dom_val")

            # if not (quad == "aug-mul-len" and (phi1[2] == 1 or phi2[2] == 1)):
            #     continue
            # if (quad != "aug-mul-len"):
            #     continue

            if a == "left":
                n_lefts_pref+=1
            if a == "right":
                n_rights_pref+=1

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

                traj2 = poi[0]
                traj2_ts_x,traj2_ts_y, pr2 =find_end_state(traj2,len(traj2)-1)
                traj2_v_s0 = board_vf[traj2[0][0]][traj2[0][1]]
                traj2_v_st = board_vf[traj2_ts_x][traj2_ts_y]
                phi2 = find_reward_features(traj2,len(traj2)-1)

                if a == "left":
                    a = "right"
                elif a == "right":
                    a = "left"


            assert ((traj2_v_st - traj2_v_s0) - (traj1_v_st - traj1_v_s0) == x)
            assert (pr2 - pr1 == y)

            er1 = pr1 + traj1_v_st - traj1_v_s0
            er2 = pr2 + traj2_v_st - traj2_v_s0

            if er1 > er2 and a != "left" or er2 > er1 and a != "right":
                incorrect+=1
            total+=1

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
                vf_r.append(pr2 - pr1)
                vf_vs0.append(traj2_v_s0 - traj1_v_s0)
                vf_vst.append(traj2_v_st - traj1_v_st)
                vf_pref.append(encoded_a)
                vf_res = add2dict(pt,encoded_a,vf_res)


            elif disp_id == "pr":
                pr_r.append(pr2 - pr1)
                pr_vs0.append(traj2_v_s0 - traj1_v_s0)
                pr_vst.append(traj2_v_st - traj1_v_st)
                pr_pref.append(encoded_a)
                pr_res = add2dict(pt,encoded_a,pr_res)


            elif disp_id == "none":
                none_r.append(pr2 - pr1)
                none_vs0.append(traj2_v_s0 - traj1_v_s0)
                none_vst.append(traj2_v_st - traj1_v_st)
                none_pref.append(encoded_a)
                none_res = add2dict(pt,encoded_a,none_res)


    print ("# of times left was preffered: " + str(n_lefts_pref))
    print ("# of times right was preffered: " + str(n_rights_pref))
    print ("# of workers: " + str(n_none_users))
    print ("\n")
    return vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref, vf_res, pr_res, none_res


def get_all_statistics_aug_human():
    quad2data1 = {"dsdt":dsdt_data,"dsst":dsst_data,"ssst":ssst_data, "sss":sss_data}
    sample_folder1 = "2021_07_29_data_samples"


    quad2data2 = {"aug-mul-len":multi_len_data, "aug-t-nt-1":t_nt_1_data, "aug-t-nt-2":t_nt_2_data}
    sample_folder2 = "2021_12_22_data_samples"


    vf_r = []
    vf_vs0 = []
    vf_vst = []
    vf_pref = []
    pr_r = []
    pr_vs0 = []
    pr_vst = []
    pr_pref = []
    none_r = []
    none_vs0 = []
    none_vst = []
    none_pref = []
    vf_res = {}
    pr_res = {}
    none_res = {}

    vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,  vf_res, pr_res, none_res = get_worker_pref_data(questions,answers,sample_folder1,quad2data1)
    vf_r2, vf_vs02, vf_vst2, vf_pref2, pr_r2, pr_vs02, pr_vst2, pr_pref2, none_r2, none_vs02, none_vst2, none_pref2,  vf_res2, pr_res2, none_res2= get_worker_pref_data(questions_aug,answers_aug,sample_folder2,quad2data2)


    vf_r.extend(vf_r2)
    vf_vs0.extend(vf_vs02)
    vf_vst.extend(vf_vst2)
    vf_pref.extend(vf_pref2)
    pr_r.extend(pr_r2)
    pr_vs0.extend(pr_vs02)
    pr_vst.extend(pr_vst2)
    pr_pref.extend(pr_pref2)
    none_r.extend(none_r2)
    none_vs0.extend(none_vs02)
    none_vst.extend(none_vst2)
    none_pref.extend(none_pref2)

    for k in vf_res2:
        if k in vf_res:
            vf_res[k].extend(vf_res2[k])
        else:
            vf_res.update({k:vf_res2[k]})

    for k in pr_res2:
        if k in pr_res:
            pr_res[k].extend(pr_res2[k])
        else:
            pr_res.update({k:pr_res2[k]})


    for k in none_res2:
        if k in none_res:
            none_res[k].extend(none_res2[k])
        else:
            none_res.update({k:none_res2[k]})

    return vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,  vf_res, pr_res, none_res


def augment_data(X,Y):
    aX = []
    ay = []
    for x,y in zip(X,Y):
        aX.append(x)
        ay.append(y)
        neg_x = []
        for val in x:
            if val != 0:
                neg_x.append(-1*val)
            else:
                neg_x.append(val)

        aX.append(neg_x)
        if y == 0:
            ay.append(1)
        elif y == 1:
            ay.append(0)
        else:
            ay.append(0.5)
    return np.array(aX), np.array(ay)


def format_y(Y):
    formatted_y= []
    n_lefts = 0
    n_rights = 0
    n_sames = 0

    for y in Y:
        if y == 0:
            n_lefts +=1
            formatted_y.append([0])
        elif y == 0.5:
            n_sames +=1
            formatted_y.append([0.5])
        elif y ==1:
            n_rights +=1
            formatted_y.append([1])
        else:
            print ("ERROR IN INPUT")
            assert False
    return torch.tensor(formatted_y,dtype=torch.float), [n_lefts/len(Y), n_rights/len(Y), n_sames/len(Y)]


def prefrence_pred_loss(output, target):
    batch_size = output.size()[0]
    output = torch.squeeze(torch.stack((output, torch.sub(1,output)),axis=2))
    output = torch.clamp(output,min=1e-35,max=None)
    output = torch.log(output)
    target = torch.squeeze(torch.stack((target, torch.sub(1,target)),axis=2))
    res = torch.mul(output,target)
    return -torch.sum(res)/batch_size

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_size, bias, prob_uniform_resp=True):
        super(LogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 1,bias=bias)
        self.c = torch.nn.Parameter(torch.tensor([0.0]))
        self.prob_uniform_resp = prob_uniform_resp
        #set initial weights to 0 for readability
        with torch.no_grad():
            self.linear1.weight = torch.nn.Parameter(torch.tensor([[0 for i in range(input_size)]],dtype=torch.float))
     def forward(self, x):
        y_pred = torch.sigmoid(self.linear1(x))
        if not self.prob_uniform_resp:
            return y_pred
        sig_c = torch.sigmoid(self.c)
        scaled_pred = torch.add(torch.mul(torch.sub(1,sig_c), y_pred),torch.divide(sig_c,2))
        return scaled_pred

def print_model_params(aX, ay,name,input_type,randomized,prob_uniform_resp):
    if input_type == "pr":
        X_train =format_X_pr(aX)
        input_size = 1
        bias = False
    elif input_type == "er":
        X_train =format_X_regret(aX)
        input_size = 1
        bias = False
    elif input_type == "full":
        X_train =format_X_full(aX)
        input_size = 3
        bias = False
    y_train,_ = format_y(ay)


    model = LogisticRegression(input_size,bias,prob_uniform_resp=prob_uniform_resp)
    # criterion = torch.nn.BCELoss()
    # criterion = PrefrenceBCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    for epoch in range(3000):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_train)
        # Compute Loss

        loss = prefrence_pred_loss(y_pred, y_train)
        # Backward pass
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        # print (name + "\n")
        if not randomized:
            for param_name, param in model.named_parameters():
                print (param_name)
                print (param)
            print ("\n")

def train_and_eval(aX, ay,name,input_type, randomized=False, prob_uniform_resp=True):
    torch.manual_seed(0)
    # print (len(aX))
    aX = np.array(aX)
    ay = np.array(ay)

    kf = KFold(n_splits=10,random_state = 0,shuffle=True)
    total_testing_log_loss = 0
    total_training_log_loss = 0
    total_random_log_loss = 0
    fold = 0

    for train_index, test_index in kf.split(aX):
        fold+=1
        X_train = aX[train_index]
        y_train = ay[train_index]
        X_test = aX[test_index]
        y_test = ay[test_index]

        if input_type == "pr":
            X_train =format_X_pr(X_train)
            input_size = 1
            bias = False
        elif input_type == "er":
            X_train =format_X_regret(X_train)
            input_size = 1
            bias = False
        elif input_type == "full":
            X_train =format_X_full(X_train)
            input_size = 3
            bias = False
        else:
            input_size = 1
            bias = False
        y_train,_ = format_y(y_train)

        if not randomized:
            model = LogisticRegression(input_size,bias,prob_uniform_resp=prob_uniform_resp)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
            for epoch in range(3000):
                model.train()
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(X_train)
                # Compute Loss

                # y_pred = torch.clamp(y_pred,min=1e-35,max=None)
                loss = prefrence_pred_loss(y_pred, y_train)
                # Backward pass
                loss.backward()
                optimizer.step()

            total_training_log_loss += prefrence_pred_loss(y_pred, y_train)
        else:
            random_pred = torch.tensor([[0.5] for i in range(len(y_train))],dtype=torch.float)
            total_training_log_loss +=prefrence_pred_loss(random_pred, y_train)

        with torch.no_grad():
            # print (name + "\n")
            y_test,class_probs = format_y(y_test)
            if not randomized:
                if input_type == "pr":
                    X_test =format_X_pr(X_test)
                elif input_type == "er":
                    X_test =format_X_regret(X_test)
                elif input_type == "full":
                    X_test =format_X_full(X_test)
                y_pred_test = model(X_test)
                total_testing_log_loss+=prefrence_pred_loss(y_pred_test, y_test)


            else:
                random_pred = torch.tensor([[0.5] for i in range(len(y_test))],dtype=torch.float)
                total_testing_log_loss+=prefrence_pred_loss(random_pred, y_test)


    print("%0.5f mean testing log loss" % (total_testing_log_loss/10))
    print("%0.5f mean training log loss" % (total_training_log_loss/10))
    if not randomized:
        print_model_params(aX, ay,name,input_type, randomized,prob_uniform_resp)


vf_r, vf_vs0, vf_vst, vf_pref, pr_r, pr_vs0, pr_vst, pr_pref, none_r, none_vs0, none_vst, none_pref,_,_,_  = get_all_statistics_aug_human()

print ("--------------------------------------------------------------")
print ("  Expected Return Model - Logistic Regression Test Results ")
print ("--------------------------------------------------------------")


print ("With prob of uniform response: ")
X = np.stack([none_r, none_vst, none_vs0], axis = 1)
Y = none_pref
aX, ay =  augment_data(X,Y)
train_and_eval(aX, ay,"preference_models/er_model_none_data","er", randomized=False)

print ("Without prob of uniform response: ")
X = np.stack([none_r, none_vst, none_vs0], axis = 1)
Y = none_pref
aX, ay =  augment_data(X,Y)
train_and_eval(aX, ay,"preference_models/er_model_none_data","er", randomized=False, prob_uniform_resp=False)

print ("--------------------------------------------------------------")
print ("  Partial Return Model - Logistic Regression Test Results ")
print ("--------------------------------------------------------------")

print ("With prob of uniform response: ")
X = np.stack([none_r, none_vst, none_vs0], axis = 1)
Y = none_pref
aX, ay =  augment_data(X,Y)
train_and_eval(aX, ay,"preference_models/pr_model_none_data","pr", randomized=False)

print ("Without prob of uniform response: ")
X = np.stack([none_r, none_vst, none_vs0], axis = 1)
Y = none_pref
aX, ay =  augment_data(X,Y)
train_and_eval(aX, ay,"preference_models/pr_model_none_data","pr", randomized=False, prob_uniform_resp=False)

print ("--------------------------------------------------------------")
print ("  Fully Expressed Model - Logistic Regression Test Results ")
print ("Note: The order of the input is (1) partial return, (2) end state value, and (3) start state value")
print ("--------------------------------------------------------------")

print ("With prob of uniform response: ")
X = np.stack([none_r, none_vst, none_vs0], axis = 1)
Y = none_pref
aX, ay =  augment_data(X,Y)
train_and_eval(aX, ay,"preference_models/full_model_none_data","full", randomized=False)

print ("Without prob of uniform response: ")
X = np.stack([none_r, none_vst, none_vs0], axis = 1)
Y = none_pref
aX, ay =  augment_data(X,Y)
train_and_eval(aX, ay,"preference_models/full_model_none_data","full", randomized=False, prob_uniform_resp=False)

print ("\n")

print ("--------------------------------------------------------------")
print (" Uninformed Model - Logistic Regression Test Results ")
print ("--------------------------------------------------------------")

print ("------------ Condition 3 - No info shown ------------")
X = np.stack([none_r, none_vst, none_vs0], axis = 1)
Y = none_pref
aX, ay =  augment_data(X,Y)
train_and_eval(aX, ay,"random_model","", randomized=True)
print ("\n")