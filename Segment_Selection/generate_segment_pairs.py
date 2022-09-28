import numpy as np
import random
import pickle
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
from itertools import product
import json
import codecs

def contains_cords(arr1,arr2):
    for a in arr1:
        if a[0] == arr2[0] and a[1] == arr2[1]:
            return True
    return False

def find_action_index(actions, action):
    i = 0
    for a in actions:
        if a[0] == action[0] and a[1] == action[1]:
            return i
        i+=1
    return False

def create_traj(s0_x, s0_y,action_seq,traj_length, board, rewards_function,terminal_states,blocking_cords,oneway_cords,values):
    '''
    Create a segment
    '''
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    
    t_partial_r_sum = 0

    traj = [(s0_x,s0_y)]
    x = s0_x
    y = s0_y
    step_n = 0
    is_terminal = False
    # print ("start: " + str((x,y)))
    for step1 in range(traj_length+1):
        if contains_cords(terminal_states,[x,y]):
            if step_n != traj_length:
                return False, False, False, False
            else:
                is_terminal = True
        
        if step_n != traj_length:
            a = action_seq[step1]

            traj.append(a)
            a_i = find_action_index(actions,a)
            t_partial_r_sum += rewards_function[x][y][a_i]
            if (x + a[0] >= 0 and x + a[0] < len(board) and y + a[1] >= 0 and y + a[1] < len(board)) and not contains_cords(blocking_cords,[x + a[0],  y + a[1]]):
                x = x + a[0]
                y = y + a[1]
    
        step_n+=1
    
 
    return traj, t_partial_r_sum, values[x][y],is_terminal


def check_case(condition, v_dif, anchor_point, all_dif_val2traj_space, quad):
    '''
    Check if given point in the segment space matches one of the conditions we are looking for
    '''
    eps = 0.4
    s
    if anchor_point in all_dif_val2traj_space:
        temp_quad = all_dif_val2traj_space.get(anchor_point)
        temp_quad.append(quad)
        all_dif_val2traj_space[anchor_point] = temp_quad
    else:
        all_dif_val2traj_space[anchor_point] = [quad]

    return all_dif_val2traj_space


def find_matching_trajs(board_name,traj_length=3,save_all=False):
    '''
    Find segments in the segment space that match our desired conditions
    '''
    #x is the difference of V(s_t) - V(s_0) value
    #y is the difference of partial sums value
    values = np.load("boards/" + str(board_name) + "_values.npy")
    board = np.load("boards/" + str(board_name) + "_board.npy")
    rewards_function = np.load("boards/" + str(board_name) + "_rewards_function.npy")
    terminal_states = np.load("boards/" + str(board_name) + "_terminal_states.npy")
    blocking_cords = np.load("boards/" + str(board_name) + "_blocking_cords.npy")
    oneway_cords = np.load("boards/" + str(board_name) + "_oneway_cords.npy")
    actions = [[-1,0],[1,0],[0,-1],[0,1]]


    all_seen_start_states = []
    all_traj_data = []
    all_dif_val2traj_space = {}


    # all_xs = []
    # all_ys = []
    all_xs_same_s0 = []
    all_ys_same_s0 = []
    all_seen_points_same_s0 = []

    all_xs_dif_s0 = []
    all_ys_dif_s0 = []
    all_seen_points_dif_s0 = []

    s0_xs = []
    s0_ys = []  


    #get all start states
    for i in range(len(board)):
        for j in range(len(board)):
            if not contains_cords(terminal_states,(i,j)) and not contains_cords(blocking_cords,(i,j)):
                s0_xs.append(i)
                s0_ys.append(j)
    
    all_action_seq = list(product(actions, repeat = traj_length))
    test_map = {}

    init_s_index = 0
    for s0_x,s0_y in list(zip(s0_xs,s0_ys)):
        all_seen_start_states.append((s0_x,s0_y))
        
        print ("Number of start states processed: " + str(len (all_seen_start_states)) + "/" + str((len(board)*len(board)) - len(terminal_states) - len(blocking_cords)))

        action_seq_index = 0
        for action_seq in all_action_seq:
            traj1, t1_partial_r_sum, v_t1,is_terminal = create_traj(s0_x,s0_y,action_seq,traj_length,board,rewards_function,terminal_states,blocking_cords,oneway_cords,values)

            if t1_partial_r_sum == False:
                continue

            traj_data = [traj1, t1_partial_r_sum, v_t1 - values[s0_x][s0_y],is_terminal]
            

            all_traj_data.append(traj_data)
            action_seq_index+=1
        init_s_index+=1
   
    #Find matching trajectories

    n_trajs = 0
    for traj_data1 in all_traj_data:
        print ("Number of trajectories processed: " + str(n_trajs) + "/" + str(len(all_traj_data)))
        traj1 = traj_data1[0]
        t1_partial_r_sum = traj_data1[1]
        v_dif1 = traj_data1[2]
        is_terminal1 = traj_data1[3]
        n_trajs+=1
        for traj_data2 in all_traj_data:
            traj2 = traj_data2[0]
            t2_partial_r_sum = traj_data2[1]
            v_dif2 = traj_data2[2]
            is_terminal2 = traj_data2[3]

            if traj1 == traj2:
                continue

            partial_sum_dif = t2_partial_r_sum - t1_partial_r_sum
            v_dif = v_dif2 - v_dif1
            quad = [traj1, traj2, v_dif, partial_sum_dif,(is_terminal1,is_terminal2)]

            s0_x1, s0_y1 = traj1[0]
            s0_x2, s0_y2 = traj2[0]


            if (s0_x1, s0_y1) == (s0_x2, s0_y2):
                if (v_dif, partial_sum_dif) not in all_seen_points_same_s0:
                    all_xs_same_s0.append(v_dif)
                    all_ys_same_s0.append(partial_sum_dif)
                    all_seen_points_same_s0.append((v_dif, partial_sum_dif))

            else:
                all_xs_dif_s0.append(v_dif)
                all_ys_dif_s0.append(partial_sum_dif)
                all_seen_points_dif_s0.append((v_dif, partial_sum_dif))

            if int(np.round(v_dif)) == 0:
                anchor_point = abs(partial_sum_dif)
            else:
                anchor_point = abs(np.round(v_dif))

            n_digits = 0

            #cases: (x,x), (x,-x),(x,0),
            x_corner_conds = (int(np.round(v_dif,n_digits)) == partial_sum_dif and partial_sum_dif > 0) or (int(np.round(v_dif,n_digits)) > 0 and int(np.round(v_dif,n_digits)) == -partial_sum_dif) or (int(np.round(v_dif,n_digits)) > 0 and partial_sum_dif == 0)
            all_dif_val2traj_space = check_case(x_corner_conds, v_dif, anchor_point, all_dif_val2traj_space, quad)


            #cases: (0,x), (0,0), (0,-x)
            zero_corner_conds = (int(np.round(v_dif,n_digits)) == 0 and partial_sum_dif > 0) or (int(np.round(v_dif,n_digits)) == 0 and partial_sum_dif == 0) or (int(np.round(v_dif,n_digits)) == 0 and partial_sum_dif < 0)
            all_dif_val2traj_space = check_case(zero_corner_conds, v_dif, anchor_point, all_dif_val2traj_space, quad)

            #case: (x, ~x)
            int_pts_cond = (int(np.round(v_dif,n_digits)) > 0 and np.round(partial_sum_dif,n_digits) > (anchor_point/2) and partial_sum_dif < 0)
            all_dif_val2traj_space = check_case(int_pts_cond, v_dif, anchor_point, all_dif_val2traj_space, quad)

            #case: (~0,x)
            int_pts_cond = (int(np.round(v_dif,n_digits)) < abs(partial_sum_dif) and int(np.round(v_dif,n_digits)) > 0 and partial_sum_dif < 0)
            all_dif_val2traj_space = check_case(int_pts_cond, v_dif, abs(partial_sum_dif), all_dif_val2traj_space, quad)


            #case: (x, ~0)
            int_pts_cond = (int(np.round(v_dif,n_digits)) > 0 and partial_sum_dif > -1*anchor_point and partial_sum_dif < 0)
            all_dif_val2traj_space = check_case(int_pts_cond, v_dif, anchor_point, all_dif_val2traj_space, quad)

    # # print (all_dif_val2traj_space)
    if save_all:
        pickle.dump(all_dif_val2traj_space,open("saved_data/all_dif_val2traj_space.p","wb"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    # ax.set_ylabel('Difference between traj. sum of partial returns')
    # ax.set_xlabel('Difference between traj. values V(s_t) - V(s_0)')
    # print (len(all_xs_same_s0))
    # print (len(all_xs_dif_s0))

    plt.scatter(all_xs_dif_s0, all_ys_dif_s0, color="red", alpha = 0.5)
    plt.scatter(all_xs_same_s0, all_ys_same_s0, color="blue", alpha = 0.5)
    plt.savefig('figures/all_points_found')

    find_passing_spaces(all_dif_val2traj_space)


def tuple_contains (t1,t2):
    x1,y1 = t2
    for t in t1:
        x2,y2 = t
        if (x1 == x2) and (y1 == y2):
            return True
    return False

def plot_arrs(xs,ys,name):
    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    # ax.set_ylabel('Difference between traj. sum of partial returns')
    # ax.set_xlabel('Difference between traj. values V(s_t) - V(s_0)')

    # plt.scatter(all_xs, all_ys)
    plt.scatter(xs, ys)
    plt.savefig('figures/' + name)

def find_passing_spaces(traj_dump, save_all = False):
    '''
    Find segments in the segment space that match our desired conditions
    '''
    print ("finding passing spaces....")
    passing_spaces = []
    ql_passing_spaces = {}
    ql_passing_spaces_str = {}
    sss_ql_passing_spaces_str = {} #only for points with the same start state
    ssst_ql_passing_spaces_str = {} #only for points with the same start state and terminal state
    dsst_ql_passing_spaces_str = {} #only for points with different start states and same terminal state
    all_xs = []
    all_ys = []
    all_seen_points = []

    xs_same_s0 = []
    ys_same_s0 = []
    all_points_same_s0 = []

    xs_same_s0_same_t = []
    ys_same_s0_same_t = []
    all_points_same_s0_same_t = []

    xs_dif_s0_same_t = []
    ys_dif_s0_same_t = []
    all_points_dif_s0_same_t = []

    xs_dif_s0_dif_t = []
    ys_dif_s0_dif_t = []
    all_points_dif_s0_dif_t = []



    dsdt_target_pts = [(0,-4),(0,-5),(1,1),(1,0),(2,2),(2,0),(5,0),(5,-1),(5,-2),(7,-1),(8,0),(9,-1),(10,-1),(13,-1)]
    dsst_target_pts = [(0,1),(1,0),(1,1),(2,0),(2,2),(3,0),(4,-1),(5,0),(7,0),(8,0)]
    sss_target_pts = [(0,-4),(0,-5),(1,1),(1,0),(1,-1),(1,-3),(2,2),(2,0),(5,0),(5,-1),(5,-2),(6,-1),(6,-2),(8,0),(9,-1),(11,-1)]

    possible_centers = traj_dump.get(0)
    cords_0_0 = []
    for c in possible_centers:
        if c[2] == 0 and c[3] == 0:
            cords_0_0.append(c)

    for x_val in list(traj_dump.keys()):
        quads_ = traj_dump.get(x_val)

        quads = quads_.copy()

        desired_cords = [(1,1),(1,0),(0,1),(0,-1),(1,-1)] 
        seen_desired_cords = []

        cords_1_1 = []
        cords_1_0 = []
        cords_0_1 = []
        cords_0_neg_1 =[]
        cords_1_neg_1 =[]
        interp = []

        sss_cords_1_1 = []
        sss_cords_1_0 = []
        sss_cords_0_1 = []
        sss_cords_0_neg_1 =[]
        sss_cords_1_neg_1 =[]
        sss_interp = []

        ssst_cords_1_1 = []
        ssst_cords_1_0 = []
        ssst_cords_0_1 = []
        ssst_cords_0_neg_1 =[]
        ssst_cords_1_neg_1 =[]
        ssst_interp = []

        dsst_cords_1_1 = []
        dsst_cords_1_0 = []
        dsst_cords_0_1 = []
        dsst_cords_0_neg_1 =[]
        dsst_cords_1_neg_1 =[]
        dsst_interp = []
        
        for quad in quads:
            traj1 = quad[0]
            traj2 = quad[1]
            x_val_ = quad[2]
            y_val = quad[3]
            is_term1, is_term2 = quad[4]

            if (x_val_, y_val) not in all_seen_points:
                all_xs.append(x_val_)
                all_ys.append(y_val)
                all_seen_points.append((x_val_, y_val))

            
            traj1_ts_x = traj1[0][0] + traj1[1][0] + traj1[2][0] + traj1[3][0] 
            traj1_ts_y = traj1[0][1] + traj1[1][1] + traj1[2][1] + traj1[3][1] 

            traj2_ts_x = traj2[0][0] + traj2[1][0] + traj2[2][0] + traj2[3][0] 
            traj2_ts_y = traj2[0][1] + traj2[1][1] + traj2[2][1] + traj2[3][1] 
            
            if traj1[0][0] == traj2[0][0] and traj1[0][1] == traj2[0][1] and traj1_ts_x == traj2_ts_x and traj1_ts_y == traj2_ts_y and is_term1 == True and is_term2 == True:
                if (x_val_, y_val) not in all_points_same_s0_same_t and abs(x_val_) < 40 and abs(y_val) <40:
                    xs_same_s0_same_t.append(x_val_)
                    ys_same_s0_same_t.append(y_val)
                    all_points_same_s0_same_t.append((x_val_, y_val))
                same_t = True
            else:
                same_t = False

                
            if (traj1[0][0] != traj2[0][0] or traj1[0][1] != traj2[0][1]) and traj1_ts_x == traj2_ts_x and traj1_ts_y == traj2_ts_y and is_term1 == True and is_term2 == True:
                # if (x_val_, y_val) not in dsst_target_pts:
                #     continue

                if (x_val_, y_val) not in all_points_dif_s0_same_t and abs(x_val_) < 40 and abs(y_val) <40:
                    xs_dif_s0_same_t.append(x_val_)
                    ys_dif_s0_same_t.append(y_val)
                    all_points_dif_s0_same_t.append((x_val_, y_val))
                dif_s_same_t = True
            else:
                dif_s_same_t = False



            if traj1[0][0] == traj2[0][0] and traj1[0][1] == traj2[0][1] and (traj1_ts_x != traj2_ts_x or traj1_ts_y != traj2_ts_y) and is_term1 == False and is_term2 == False:
                # if (x_val_, y_val) not in sss_target_pts:
                #     continue
                if (x_val_, y_val) not in all_points_same_s0:
                    xs_same_s0.append(x_val_)
                    ys_same_s0.append(y_val)
                    all_points_same_s0.append((x_val_, y_val))
                same_s0 = True
            else:
                same_s0 = False

            if not same_s0 and not dif_s_same_t and not same_t:
                # if (x_val_, y_val) not in dsdt_target_pts:
                #     continue
                if (x_val_, y_val) not in all_points_dif_s0_dif_t and abs(x_val_) < 20 and abs(y_val) <20:
                    xs_dif_s0_dif_t.append(x_val_)
                    ys_dif_s0_dif_t.append(y_val)
                    all_points_dif_s0_dif_t.append((x_val_, y_val))

          

            #make sure we do not remove desired coordinates because of interpolated points
            if abs(np.round(x_val_)) == x_val or abs(np.round(y_val)) == x_val:
                x_val_mapped = int(np.sign(x_val_))
                y_val_mapped = int(np.sign(y_val))
               
                if (x_val_mapped,y_val_mapped) in desired_cords:
                   
                    #interpolated points like (2,-8)
                    if (x_val_mapped,y_val_mapped) == (1,-1) and not(abs(np.round(x_val_)) == x_val and abs(y_val) == x_val):
                        interp.append(quad)
                        if (same_s0):
                            sss_interp.append(quad)
                        if (same_t):
                            ssst_interp.append(quad)
                        if (dif_s_same_t):
                            dsst_interp.append(quad)

                        continue
                    elif (x_val_mapped,y_val_mapped) not in seen_desired_cords:
                        seen_desired_cords.append((x_val_mapped,y_val_mapped))

                    if (x_val_mapped,y_val_mapped) == (1,1):
                        cords_1_1.append(quad)
                        if (same_s0):
                            sss_cords_1_1.append(quad)
                        if (same_t):
                            ssst_cords_1_1.append(quad)
                        if (dif_s_same_t):
                            dsst_cords_1_1.append(quad)
                    elif (x_val_mapped,y_val_mapped) == (1,0):
                        cords_1_0.append(quad)
                        if (same_s0):
                            sss_cords_1_0.append(quad)
                        if (same_t):
                            ssst_cords_1_0.append(quad)
                        if (dif_s_same_t):
                            dsst_cords_1_0.append(quad)
                    elif (x_val_mapped,y_val_mapped) == (0,1):
                        cords_0_1.append(quad)
                        if (same_s0):
                            sss_cords_0_1.append(quad)
                        if (same_t):
                            ssst_cords_0_1.append(quad)
                        if (dif_s_same_t):
                            dsst_cords_0_1.append(quad)
                    elif (x_val_mapped,y_val_mapped) == (0,-1):
                        cords_0_neg_1.append(quad)
                        if (same_s0):
                            sss_cords_0_neg_1.append(quad)
                        if (same_t):
                            ssst_cords_0_neg_1.append(quad)
                        if (dif_s_same_t):
                            dsst_cords_0_neg_1.append(quad)
                    elif (x_val_mapped,y_val_mapped) == (1,-1):
                        cords_1_neg_1.append(quad)
                        if (same_s0):
                            sss_cords_1_neg_1.append(quad)
                        if (same_t):
                            ssst_cords_1_neg_1.append(quad)
                        if (dif_s_same_t):
                            dsst_cords_1_neg_1.append(quad)
                else:
                    interp.append(quad)
                    if (same_s0):
                        sss_interp.append(quad)
                    if (same_t):
                        ssst_interp.append(quad)
                    if (dif_s_same_t):
                            dsst_interp.append(quad)
    
            
        #------------------------------------------Add points with same start state------------------------------------------
        if (len(sss_cords_1_1) < 100):
            sss_cords_1_1_passing = sss_cords_1_1
        else:
            sss_cords_1_1_passing = random.choices(sss_cords_1_1,k=100)
        if (len(sss_cords_1_0) < 100):
            sss_cords_1_0_passing = sss_cords_1_0
        else:
            sss_cords_1_0_passing = random.choices(sss_cords_1_0,k=100)
        if (len(sss_cords_0_1) < 100):
            sss_cords_0_1_passing = sss_cords_0_1
        else:
            sss_cords_0_1_passing = random.choices(sss_cords_0_1,k=100)
        if (len(sss_cords_0_neg_1) < 100):
            sss_cords_0_neg_1_passing = sss_cords_0_neg_1
        else:
            sss_cords_0_neg_1_passing = random.choices(sss_cords_0_neg_1,k=100)
        if (len(sss_cords_1_neg_1) < 100):
            sss_cords_1_neg_1_passing = sss_cords_1_neg_1
        else:
            sss_cords_1_neg_1_passing = random.choices(sss_cords_1_neg_1,k=100) 
        if len (sss_interp) < 100:
            sss_interp_passing = sss_interp
        else:
            sss_interp_passing = random.choices(sss_interp,k=100) 
        if len(sss_cords_1_1_passing) > 0:
            sss_ql_passing_spaces_str[str((x_val,x_val))] = sss_cords_1_1_passing
        if len(sss_cords_1_0_passing) > 0:
            sss_ql_passing_spaces_str[str((x_val,0))] = sss_cords_1_0_passing
        if len(sss_cords_0_1_passing) > 0:
            sss_ql_passing_spaces_str[str((0,x_val))] = sss_cords_0_1_passing
        if len(sss_cords_0_neg_1_passing) > 0:
            sss_ql_passing_spaces_str[str((0,-1*x_val))] = sss_cords_0_neg_1_passing
        if len(sss_cords_1_neg_1_passing) > 0:
            sss_ql_passing_spaces_str[str((x_val,-1*x_val))] = sss_cords_1_neg_1_passing
        
        #add all interpolated points seperatley
        # if len(sss_interp_passing) > 100:
        for interp_pt in sss_interp_passing:
            interp_x_val = interp_pt[2]
            interp_y_val = interp_pt[3]
            key = str((interp_x_val,interp_y_val))
            if (key in sss_ql_passing_spaces_str):
                out = sss_ql_passing_spaces_str.get(key)
                out.append(interp_pt)
                sss_ql_passing_spaces_str[key] = out
            else:
                sss_ql_passing_spaces_str[key] = [interp_pt]

        #------------------------------------------Add points with same start state and same terminal state------------------------------------------
        if (len(ssst_cords_1_1) < 100):
            ssst_cords_1_1_passing = ssst_cords_1_1
        else:
            ssst_cords_1_1_passing = random.choices(ssst_cords_1_1,k=100)
        if (len(ssst_cords_1_0) < 100):
            ssst_cords_1_0_passing = ssst_cords_1_0
        else:
            ssst_cords_1_0_passing = random.choices(ssst_cords_1_0,k=100)
        if (len(ssst_cords_0_1) < 100):
            ssst_cords_0_1_passing = ssst_cords_0_1
        else:
            ssst_cords_0_1_passing = random.choices(ssst_cords_0_1,k=100)
        if (len(ssst_cords_0_neg_1) < 100):
            ssst_cords_0_neg_1_passing = ssst_cords_0_neg_1
        else:
            ssst_cords_0_neg_1_passing = random.choices(ssst_cords_0_neg_1,k=100)
        if (len(ssst_cords_1_neg_1) < 100):
            ssst_cords_1_neg_1_passing = ssst_cords_1_neg_1
        else:
            ssst_cords_1_neg_1_passing = random.choices(ssst_cords_1_neg_1,k=100) 
        if len (ssst_interp) < 100:
            ssst_interp_passing = ssst_interp
        else:
            ssst_interp_passing = random.choices(ssst_interp,k=100) 


        if len(ssst_cords_1_1_passing) > 0:
            ssst_ql_passing_spaces_str[str((x_val,x_val))] = ssst_cords_1_1_passing
        if len(ssst_cords_1_0_passing) > 0:
            ssst_ql_passing_spaces_str[str((x_val,0))] = ssst_cords_1_0_passing
        if len(ssst_cords_0_1_passing) > 0:
            ssst_ql_passing_spaces_str[str((0,x_val))] = ssst_cords_0_1_passing
        if len(ssst_cords_0_neg_1_passing) > 0:
            ssst_ql_passing_spaces_str[str((0,-1*x_val))] = ssst_cords_0_neg_1_passing
        if len(ssst_cords_1_neg_1_passing) > 0:
            ssst_ql_passing_spaces_str[str((x_val,-1*x_val))] = ssst_cords_1_neg_1_passing
        
        #add all interpolated points seperatley
        if len(ssst_interp_passing) > 0:
            for interp_pt in ssst_interp_passing:
                interp_x_val = interp_pt[2]
                interp_y_val = interp_pt[3]
                key = str((interp_x_val,interp_y_val))
                if (key in ssst_ql_passing_spaces_str):
                    out = ssst_ql_passing_spaces_str.get(key)
                    out.append(interp_pt)
                    ssst_ql_passing_spaces_str[key] = out
                else:
                    ssst_ql_passing_spaces_str[key] = [interp_pt]
        #------------------------------------------Add points with different start states and same terminal state------------------------------------------
        if (len(dsst_cords_1_1) < 100):
            dsst_cords_1_1_passing = dsst_cords_1_1
        else:
            dsst_cords_1_1_passing = random.choices(dsst_cords_1_1,k=100)
        if (len(dsst_cords_1_0) < 100):
            dsst_cords_1_0_passing = dsst_cords_1_0
        else:
            dsst_cords_1_0_passing = random.choices(dsst_cords_1_0,k=100)
        if (len(dsst_cords_0_1) < 100):
            dsst_cords_0_1_passing = dsst_cords_0_1
        else:
            dsst_cords_0_1_passing = random.choices(dsst_cords_0_1,k=100)
        if (len(dsst_cords_0_neg_1) < 100):
            dsst_cords_0_neg_1_passing = dsst_cords_0_neg_1
        else:
            dsst_cords_0_neg_1_passing = random.choices(dsst_cords_0_neg_1,k=100)
        if (len(dsst_cords_1_neg_1) < 100):
            dsst_cords_1_neg_1_passing = dsst_cords_1_neg_1
        else:
            dsst_cords_1_neg_1_passing = random.choices(dsst_cords_1_neg_1,k=100) 
        if len (dsst_interp) < 100:
            dsst_interp_passing = dsst_interp
        else:
            dsst_interp_passing = random.choices(dsst_interp,k=100) 


        if len(dsst_cords_1_1_passing) > 0:
            dsst_ql_passing_spaces_str[str((x_val,x_val))] = dsst_cords_1_1_passing
        if len(ssst_cords_1_0_passing) > 0:
            dsst_ql_passing_spaces_str[str((x_val,0))] = dsst_cords_1_0_passing
        if len(ssst_cords_0_1_passing) > 0:
            dsst_ql_passing_spaces_str[str((0,x_val))] = dsst_cords_0_1_passing
        if len(ssst_cords_0_neg_1_passing) > 0:
            dsst_ql_passing_spaces_str[str((0,-1*x_val))] = dsst_cords_0_neg_1_passing
        if len(ssst_cords_1_neg_1_passing) > 0:
            dsst_ql_passing_spaces_str[str((x_val,-1*x_val))] = dsst_cords_1_neg_1_passing
        
        #add all interpolated points seperatley
        if len(dsst_interp_passing) > 0:
            for interp_pt in dsst_interp_passing:
                interp_x_val = interp_pt[2]
                interp_y_val = interp_pt[3]
                key = str((interp_x_val,interp_y_val))
                if (key in dsst_ql_passing_spaces_str):
                    out = dsst_ql_passing_spaces_str.get(key)
                    out.append(interp_pt)
                    dsst_ql_passing_spaces_str[key] = out
                else:
                    dsst_ql_passing_spaces_str[key] = [interp_pt]
        #------------------------------------------Add any points that fit our criteria------------------------------------------

        # if len(seen_desired_cords) == 5:
        if True:
            print ("anchor: " + str(x_val))

            quads.extend(cords_0_0)
            passing_spaces.append(quads)

            if (len(cords_1_1) > 100):
                cords_1_1_passing = random.choices(cords_1_1,k=100)
            else:
                cords_1_1_passing = cords_1_1
            
            if (len(cords_1_0) > 100):
                cords_1_0_passing = random.choices(cords_1_0,k=100)
            else:
                cords_1_0_passing = cords_1_0

            if (len(cords_0_1) > 100):
                cords_0_1_passing = random.choices(cords_0_1,k=100)
            else:
                cords_0_1_passing = cords_0_1
            
            if (len(cords_0_neg_1) > 100):
                cords_0_neg_1_passing = random.choices(cords_0_neg_1,k=100)
            else:
                cords_0_neg_1_passing = cords_0_neg_1
            
            if (len(cords_1_neg_1) > 100):
                cords_1_neg_1_passing = random.choices(cords_1_neg_1,k=100)
            else:
                cords_1_neg_1_passing = cords_1_1
            
            if (len(cords_0_0) > 100):
                cords_0_0_passing = random.choices(cords_0_0,k=100)
            else:
                cords_0_0_passing = cords_0_0

            if len (interp) < 100:
                interp_passing = interp
            else:
                interp_passing = random.choices(interp,k=100) 

            ql_passing_spaces[(x_val,x_val)] = cords_1_1_passing
            ql_passing_spaces[(x_val,0)] = cords_1_0_passing
            ql_passing_spaces[(0,x_val)] = cords_0_1_passing
            ql_passing_spaces[(0,-1*x_val)] = cords_0_neg_1_passing
            ql_passing_spaces[(x_val,-1*x_val)] = cords_1_neg_1_passing
            ql_passing_spaces[(0,0)] = cords_0_0_passing
            ql_passing_spaces[str(x_val) + "_between_pts"] = interp_passing

            # ql_passing_spaces_str[str((x_val,x_val))] = cords_1_1_passing
            # ql_passing_spaces_str[str((x_val,0))] = cords_1_0_passing
            # ql_passing_spaces_str[str((0,x_val))] = cords_0_1_passing
            # ql_passing_spaces_str[str((0,-1*x_val))] = cords_0_neg_1_passing
            # ql_passing_spaces_str[str((x_val,-1*x_val))] = cords_1_neg_1_passing
            # ql_passing_spaces_str[str((0,0))] = cords_0_0_passing
            # ql_passing_spaces_str[str(x_val) + "_between_pts"] = interp_passing

            if len(cords_1_1_passing) > 0:
                ql_passing_spaces_str[str((x_val,x_val))] = cords_1_1_passing
            if len(cords_1_0_passing) > 0:
                ql_passing_spaces_str[str((x_val,0))] = cords_1_0_passing
            if len(cords_0_1_passing) > 0:
                ql_passing_spaces_str[str((0,x_val))] = cords_0_1_passing
            if len(cords_0_neg_1_passing) > 0:
                ql_passing_spaces_str[str((0,-1*x_val))] = cords_0_neg_1_passing
            if len(cords_1_neg_1_passing) > 0:
                ql_passing_spaces_str[str((x_val,-1*x_val))] = cords_1_neg_1_passing  
            if len(cords_0_0_passing) > 0:
                ql_passing_spaces_str[str((0,0))] = cords_0_0_passing
            
            if len(interp_passing) > 0:
                for interp_pt in interp_passing:
                    interp_x_val = interp_pt[2]
                    interp_y_val = interp_pt[3]
                    key = str((interp_x_val,interp_y_val))
                    if (key in ql_passing_spaces_str):
                        out = ql_passing_spaces_str.get(key)
                        out.append(interp_pt)
                        ql_passing_spaces_str[key] = out
                    else:
                        ql_passing_spaces_str[key] = [interp_pt]



   

    if save_all:
        np.save("saved_data/passing_spaces.npy",passing_spaces)
        
    # pickle.dump(ql_passing_spaces,open("saved_data/ql_passing_spaces.p","wb"))
    json.dump(ql_passing_spaces_str,codecs.open("saved_data/2021_07_29_dsdt.json", 'w', encoding='utf-8'))
    json.dump(sss_ql_passing_spaces_str,codecs.open("saved_data/2021_07_29_sss.json", 'w', encoding='utf-8'))
    json.dump(ssst_ql_passing_spaces_str,codecs.open("saved_data/2021_07_29_ssst.json", 'w', encoding='utf-8'))
    json.dump(dsst_ql_passing_spaces_str,codecs.open("saved_data/2021_07_29_dsst.json", 'w', encoding='utf-8'))

    # return passing_spaces
    plot_arrs(xs_same_s0_same_t,ys_same_s0_same_t,"all_pts_ssst")
    plot_arrs(xs_dif_s0_same_t,ys_dif_s0_same_t,"all_pts_dsst")
    plot_arrs(xs_dif_s0_dif_t,ys_dif_s0_dif_t,"all_pts_dsdt")
    plot_arrs(xs_same_s0,ys_same_s0,"all_pts_sss")


def process_passing_spaces(passing_spaces):
    print ("processing passing spaces....")
    n_spaces = 0
    print (len(passing_spaces))
    for quads in passing_spaces:
        xs_same_s0 = []
        ys_same_s0 = []
        all_points_same_s0 = []

        xs_dif_s0 = []
        ys_dif_s0 = []
        all_points_dif_s0 = []
        for q in quads:
            traj1_s0 = q[0][0]
            x1,y1 = traj1_s0

            traj2_s0 = q[1][0]
            x2,y2 = traj2_s0

            if x1 == x2 and y1 == y2:
                if (q[2],q[3]) not in all_points_same_s0:
                    xs_same_s0.append(q[2])
                    ys_same_s0.append(q[3])
                    all_points_same_s0.append((q[2], q[3]))
            else:
                if (q[2],q[3]) not in all_points_dif_s0:
                    xs_dif_s0.append(q[2])
                    ys_dif_s0.append(q[3])
                    all_points_dif_s0.append((q[2], q[3]))


        fig = plt.figure()
        # ax1 = fig.add_subplot(121,projection='3d')
        # # plt.axes(projection='3d')
        # # ax.set_ylabel('Difference between traj. sum of partial returns')
        # # ax.set_xlabel('Difference between traj. values V(s_t) - V(s_0)')
        # ax1.scatter3D(xs, ys, zs,cmap='Greens')

        # ax2 = fig.add_subplot(122)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        
        plt.scatter(xs_dif_s0,ys_dif_s0,color="red", alpha=0.5)
        plt.scatter(xs_same_s0,ys_same_s0,color="blue", alpha=0.5)

        plt.savefig('figures/n_space_' + str(n_spaces))
        n_spaces +=1


# find_matching_trajs("test_single_goal_mud",traj_length=3)

find_matching_trajs("2021-07-29_sparseboard2-notrap",traj_length=3)
