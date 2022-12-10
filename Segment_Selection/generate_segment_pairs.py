import numpy as np
import pickle
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
from itertools import product
import json
import codecs
import random


def find_action_index(action):
    '''
    Finds the index of an action (represented as an array of size 2) given an array of all actions

    Input:
    - action: an action, represented as an array of size 2 [x displacement, y displacement]

    Output:
    - the index of the action in the list of actions
    '''

    actions = {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}
    return actions[tuple(action)]


def create_seg(s0_x, s0_y,action_seq,seg_length, board, rewards_function,terminal_states,blocking_cords,oneway_cords,values):
    '''
    Creates a segment

    Input:
    - s0_x, s0_y: the start state coordinates
    - action_seq: the sequence of actions in the segment
    - seg_length: the segment length
    - board: the MDP
    - rewards_function: the reward function for the MDP
    - terminal_states: a list of all terminal states in the MDP
    - blocking_cords: a list of all inaccessible states in the MDP
    - oneway_cords: any coordinates in the MDP where the agent can only move in one direction (depreciated)
    - values: the value function for the MDP

    Output:
    seg: a segment, represented as it's start state and the corrosponding action sequence
    t_partial_r_sum: the segments partial return
    values[x][y]: the segment's end state value
    is_terminal: whether or not the segment ends in a terminal state
    '''
    t_partial_r_sum = 0

    seg = [(s0_x,s0_y)]
    x = s0_x
    y = s0_y
    step_n = 0
    is_terminal = False
    # print ("start: " + str((x,y)))


    for step1 in range(seg_length+1):
        if (x,y) in terminal_states:
            if step_n != seg_length:
                return False, False, False, False, False
            else:
                is_terminal = True

        if step_n != seg_length:
            a = action_seq[step1]

            seg.append(a)
            a_i = find_action_index(a)
            t_partial_r_sum += rewards_function[x][y][a_i]
            if (x + a[0] >= 0 and x + a[0] < len(board) and y + a[1] >= 0 and y + a[1] < len(board)) and (x + a[0],  y + a[1]) not in blocking_cords:
                x = x + a[0]
                y = y + a[1]

        step_n+=1

    return seg, t_partial_r_sum, values[x][y],is_terminal, (x,y)

def subsample_segment_pairs(seg_dict):
    '''
    Ensures that there are at most 100 segment pairs per coordinate in the given dictionary

    Input:
    - seg_dict: a dictionary mapping coordinates in the segment space to a list of segment pairs

    Output:
    - subsampled_seg_dict: a dictionary mapping coordinates in the segment space to a list of segment pairs, where there are at most 100 segment pairs
    '''

    subsampled_seg_dict = {}
    for key in seg_dict:
        subsampled_seg_dict[key] = random.sample(seg_dict[key], min(len(seg_dict[key]),100))
    return subsampled_seg_dict



def find_matching_segs(board_name,seg_length=3):
    '''
    This function searches the entire segment space to find segment pairs that match our desired criteria.
    The segment space x-axis is the difference in the change in state value (V(s_t) - V(s_0)) between two segments.
    The segment space y-axis is the different in partial return between two segments.

    Here, we are looking for segments at specific coordinates in the segment space (defined below) that have:
    - different start states and different end states
    - different start states and the same end states
    - the same start state and different end states
    - the same start and end states

    Input:
    - board_name: the name of the MDP file, which is found in the boards subdirectory
    - seg_length: the length of the segments (measured as number of transitions) being collected

    Output
    - sss_ql_passing_spaces: a dictionary storing points and their corrosponding segment pairs (a maximum of 100) with the same start state and different end state
    - ssst_ql_passing_spaces: a dictionary storing points and their corrosponding segment pairs (a maximum of 100) with the same start state and end state
    - dsst_ql_passing_spaces: a dictionary storing points and their corrosponding segment pairs (a maximum of 100) with different start states and same end state
    - dsdt_ql_passing_spaces: a dictionary storing points and their corrosponding segment pairs (a maximum of 100) with different start states and different end states
    '''

    values = np.load("boards/" + str(board_name) + "_values.npy")
    board = np.load("boards/" + str(board_name) + "_board.npy")
    rewards_function = np.load("boards/" + str(board_name) + "_rewards_function.npy")
    terminal_states = np.load("boards/" + str(board_name) + "_terminal_states.npy")
    blocking_cords = np.load("boards/" + str(board_name) + "_blocking_cords.npy")
    oneway_cords = np.load("boards/" + str(board_name) + "_oneway_cords.npy")
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    terminal_states = set([tuple(state) for state in terminal_states])
    blocking_cords = set([tuple(state) for state in blocking_cords])


    all_seen_start_states = []
    all_seg_data = []

    s0_xs = []
    s0_ys = []

    #the coordinates we want in our segment space for segment pairs with the different start states and different end states
    dsdt_target_pts = [(0,-4),(0,-5),(1,1),(1,0),(2,2),(2,0),(5,0),(5,-1),(5,-2),(7,-1),(8,0),(9,-1),(10,-1),(13,-1)]

    #the coordinates we want in our segment space for segment pairs with different start states and the same end states
    dsst_target_pts = [(0,1),(1,0),(1,1),(2,0),(2,2),(3,0),(4,-1),(5,0),(7,0),(8,0)]

    #the coordinates we want in our segment space for segment pairs with the same start state and different end states
    sss_target_pts = [(0,-4),(0,-5),(1,1),(1,0),(1,-1),(1,-3),(2,2),(2,0),(5,0),(5,-1),(5,-2),(6,-1),(6,-2),(8,0),(9,-1),(11,-1)]

    #Note: Because of the lack of coordinates that match this condition, we want to find all segment pairs that have the same start and end states

    sss_ql_passing_spaces = {} #store points with the same start state and different end state
    ssst_ql_passing_spaces = {} #store points with the same start state and end state
    dsst_ql_passing_spaces = {} #store points with different start states and same end state
    dsdt_ql_passing_spaces = {} #store points with different start states and different end states


    #get all start states
    for i in range(len(board)):
        for j in range(len(board)):
            if (i,j) not in terminal_states and (i,j) not in blocking_cords:
                s0_xs.append(i)
                s0_ys.append(j)

    #generate all possible action sequences
    all_action_seq = list(product(actions, repeat = seg_length))
    test_map = {}

    init_s_index = 0
    #build all possible segments
    for s0_x,s0_y in list(zip(s0_xs,s0_ys)):
        all_seen_start_states.append((s0_x,s0_y))
        print ("Number of start states processed: " + str(len (all_seen_start_states)) + "/" + str((len(board)*len(board)) - len(terminal_states) - len(blocking_cords)))
        action_seq_index = 0
        for action_seq in all_action_seq:
            seg1, t1_partial_r_sum, v_t1,is_terminal, end_cords = create_seg(s0_x,s0_y,action_seq,seg_length,board,rewards_function,terminal_states,blocking_cords,oneway_cords,values)
            if t1_partial_r_sum == False:
                continue
            seg_data = [seg1, t1_partial_r_sum, v_t1 - values[s0_x][s0_y],is_terminal, end_cords]
            all_seg_data.append(seg_data)
            action_seq_index+=1
        init_s_index+=1

    # assert False
    #find segment pairs that match our search criteria
    n_segs = 0
    for seg_data1 in all_seg_data:
        print ("Number of segments processed: " + str(n_segs) + "/" + str(len(all_seg_data)))
        seg1 = seg_data1[0]
        t1_partial_r_sum = seg_data1[1]
        v_dif1 = seg_data1[2]
        is_terminal1 = seg_data1[3]
        end_cords1 = seg_data1[4]
        n_segs+=1


        for seg_data2 in all_seg_data:
            seg2 = seg_data2[0]
            t2_partial_r_sum = seg_data2[1]
            v_dif2 = seg_data2[2]
            is_terminal2 = seg_data2[3]
            end_cords2 = seg_data2[4]

            if seg1 == seg2:
                continue

            partial_sum_dif = t2_partial_r_sum - t1_partial_r_sum
            v_dif = v_dif2 - v_dif1
            quad = [seg1, seg2, v_dif, partial_sum_dif,(is_terminal1,is_terminal2)]

            #get each segments start state
            s0_x1, s0_y1 = seg1[0]
            s0_x2, s0_y2 = seg2[0]

            #get each segments end state
            seg1_ts_x = end_cords1[0]
            seg1_ts_y = end_cords1[1]

            seg2_ts_x = end_cords2[0]
            seg2_ts_y = end_cords2[1]

            #found segment pair with the same start and end states
            if seg1[0][0] == seg2[0][0] and seg1[0][1] == seg2[0][1] and seg1_ts_x == seg2_ts_x and seg1_ts_y == seg2_ts_y:

                if str((v_dif, partial_sum_dif)) not in ssst_ql_passing_spaces:
                    ssst_ql_passing_spaces[str((v_dif, partial_sum_dif))] = [quad]
                else:
                    ssst_ql_passing_spaces[str((v_dif, partial_sum_dif))].append(quad)
            #found segment pair with the same start and different end states
            elif seg1[0][0] == seg2[0][0] and seg1[0][1] == seg2[0][1] and not (seg1_ts_x == seg2_ts_x and seg1_ts_y == seg2_ts_y):
                if (v_dif, partial_sum_dif) not in sss_target_pts:
                    continue

                if str((v_dif, partial_sum_dif)) not in sss_ql_passing_spaces:
                    sss_ql_passing_spaces[str((v_dif, partial_sum_dif))] = [quad]
                else:
                    sss_ql_passing_spaces[str((v_dif, partial_sum_dif))].append(quad)
            #found segment pair with the different start and different end states
            elif not (seg1[0][0] == seg2[0][0] and seg1[0][1] == seg2[0][1]) and not (seg1_ts_x == seg2_ts_x and seg1_ts_y == seg2_ts_y):
                if (v_dif, partial_sum_dif) not in dsdt_target_pts:
                    continue

                if str((v_dif, partial_sum_dif)) not in dsdt_ql_passing_spaces:
                    dsdt_ql_passing_spaces[str((v_dif, partial_sum_dif))] = [quad]
                else:
                    dsdt_ql_passing_spaces[str((v_dif, partial_sum_dif))].append(quad)
            #found segment pair with the different start and same end states
            elif not (seg1[0][0] == seg2[0][0] and seg1[0][1] == seg2[0][1]) and (seg1_ts_x == seg2_ts_x and seg1_ts_y == seg2_ts_y):
                if (v_dif, partial_sum_dif) not in dsst_target_pts:
                    continue
                if str((v_dif, partial_sum_dif)) not in dsst_ql_passing_spaces:
                    dsst_ql_passing_spaces[str((v_dif, partial_sum_dif))] = [quad]
                else:
                    dsst_ql_passing_spaces[str((v_dif, partial_sum_dif))].append(quad)

    ssst_ql_passing_spaces = subsample_segment_pairs(ssst_ql_passing_spaces)
    sss_ql_passing_spaces = subsample_segment_pairs(sss_ql_passing_spaces)
    dsdt_ql_passing_spaces = subsample_segment_pairs(dsdt_ql_passing_spaces)
    dsst_ql_passing_spaces = subsample_segment_pairs(dsst_ql_passing_spaces)

    json.dump(ssst_ql_passing_spaces,codecs.open("saved_data/2021_07_29_ssst.json", 'w', encoding='utf-8'))
    json.dump(sss_ql_passing_spaces,codecs.open("saved_data/2021_07_29_sss.json", 'w', encoding='utf-8'))
    json.dump(dsdt_ql_passing_spaces,codecs.open("saved_data/2021_07_29_dsdt.json", 'w', encoding='utf-8'))
    json.dump(dsst_ql_passing_spaces,codecs.open("saved_data/2021_07_29_dsst.json", 'w', encoding='utf-8'))

find_matching_segs("2021-07-29_sparseboard2-notrap",seg_length=3)
