import numpy as np
import random
from rl_algos import value_iteration, get_gt_avg_return,build_pi,iterative_policy_evaluation,learn_successor_feature_iter,build_random_policy
import pickle
import math
from sklearn.preprocessing import minmax_scale
from prettytable import PrettyTable
import cv2
import matplotlib.pyplot as plt
from scipy import stats

from rl_algos import value_iteration, get_gt_avg_return,build_pi,build_pi_from_feats,build_pi_from_nn_feats,iterative_policy_evaluation,learn_successor_feature_iter,build_random_policy

def contains_cords(arr1,arr2):
    for a in arr1:
        if a[0] == arr2[0] and a[1] == arr2[1]:
            return True
    return False

#102 - recovered optimal policy
#105 - failed to recover better than random policy; scaled return following learned policy: -1.9199553856744647
num_prefs = 10000
learn_oaf = True
preference_model = "regret"
preference_assum = "pr"
mode = "deterministic"
use_extended_SF = True
generalize_SF = False

t_mdp = PrettyTable(['MDP', "Mean shift", "Shift var", "Mean rescale multiplier", "Rescale multiplier var"])

for trial in range(100,130):
    fp =  "MDP_" + str(trial) + "_" + str(learn_oaf) + "_" + preference_model + "_" + preference_assum + "_mode=" + mode + "_extended_SF=" + str(use_extended_SF)  + "_generalize_SF=" + str(generalize_SF) + "_num_prefs=" + str(num_prefs)
    pred_OAF = np.load("test_policy_learning/" + fp + "_OAF.npy")

    #pred_OAF = np.load("test_policy_learning/MDP_" + str(trial) + "_OAF.npy")
    gt_OAF = np.load("test_policy_learning/MDP_" + str(trial) + "_gt_OAF.npy")

    gt_rew_vec = np.load("random_MDPs/MDP_" + str(trial) +"gt_rew_vec.npy",mmap_mode="r")
    with open("random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
        env = pickle.load(rf)    
        env.generate_transition_probs()
        env.set_custom_reward_function(gt_rew_vec)
        env.get_blocking_cords()
        env.get_terminal_cords()


    pred_OAF = pred_OAF.reshape((env.height, env.width, 4))
    gt_OAF = gt_OAF.reshape((env.height, env.width, 4))

    t = PrettyTable(['(state, action)', 'Predicted optimal adv.', "GT optimal adv.", "Shifted adv.", "Norm shifted adv.", "|Norm shifted adv. - GT adv.|", "Same optimal action?"])

    shifts = []
    rescalers = []
    for i in range (env.height):
        for j in range (env.width):
            if max(pred_OAF[i][j]) > 0:
                shift = -max(pred_OAF[i][j])
            elif max(pred_OAF[i][j]) < 0:
                shift = -(max(pred_OAF[i][j]))

            shifts.append(-shift)

            target_mean_adv = np.mean(gt_OAF[i][j])
            current_mean_adv = np.mean([pred_OAF[i][j][a_i] + shift for a_i in range(4)])

            rescalers.append(target_mean_adv/current_mean_adv)

    t_mdp.add_row([trial, np.mean(shifts), np.var(shifts), np.mean(rescalers), np.var(rescalers)])


    # shifts = []
    # rescalers = []
    rounded_plot_xs = []
    rounded_plot_ys = []

    plot_xs = []
    plot_ys = []

    seen_cords = {}
    for i in range (env.height):
        for j in range (env.width):
            if max(pred_OAF[i][j]) > 0:
                shift = -max(pred_OAF[i][j])
            elif max(pred_OAF[i][j]) < 0:
                shift = -(max(pred_OAF[i][j]))


            target_mean_adv = np.mean(gt_OAF[i][j])
            current_mean_adv = np.mean([pred_OAF[i][j][a_i] + shift for a_i in range(4)])

            
            gt_oas = np.argwhere(gt_OAF[i][j] == np.amax(gt_OAF[i][j]))
            pred_oas = np.argwhere(pred_OAF[i][j] == np.amax(pred_OAF[i][j]))

            

            is_same_opt = any(np.isin(gt_oas,pred_oas))

            for a_i in range(4):
                if contains_cords(env.blocking_cords, [i,j]) or contains_cords(env.terminal_cords, [i,j]):
                    continue
                diff = abs(pred_OAF[i][j][a_i] - gt_OAF[i][j][a_i])

                x_cord = np.round(gt_OAF[i][j][a_i],1)
                y_cord = np.round(pred_OAF[i][j][a_i],1)
                rounded_plot_xs.append(x_cord)
                rounded_plot_ys.append(y_cord)

                plot_xs.append(gt_OAF[i][j][a_i])
                plot_ys.append(pred_OAF[i][j][a_i])

                if (x_cord, y_cord) in seen_cords:
                    seen_cords[(x_cord, y_cord)] +=1
                else:
                    seen_cords[(x_cord, y_cord)] =1
                
            
                t.add_row([str(((i,j), a_i)), str(np.round(pred_OAF[i][j][a_i],3)), str(np.round(gt_OAF[i][j][a_i],3)), str(np.round(pred_OAF[i][j][a_i] + shift,3)), str(np.round((pred_OAF[i][j][a_i] + shift)*target_mean_adv/current_mean_adv, 3)),np.round(((pred_OAF[i][j][a_i] + shift)*target_mean_adv/current_mean_adv) - gt_OAF[i][j][a_i], 3) , is_same_opt])


    # print(t)
    # print (trial)
    f = open("analysis/" + fp + "_analysis.txt", "w")
    f.write(str(t))
    f.close()
    
    plt.figure(trial)

    s = []
    for x,y in zip(rounded_plot_xs, rounded_plot_ys):
        s.append(seen_cords[(x,y)]*5)
    # assert False

    corr, p_val = stats.spearmanr(plot_xs, plot_ys)
    print ("MDP " + str(trial))
    print ("Correlation between the predicted advantage estimate and the ground truth: " + str(corr))
    print ("P val: " + str(p_val))
    print ("\n")


    plt.scatter(rounded_plot_xs, rounded_plot_ys,s=s)
    plt.savefig("analysis_plots/MDP_" + str(trial) + "_advantage_plot.png")
    plt.close(trial)



f = open("analysis/" + fp + "_all_MDPs_analysis.txt", "w")
f.write(str(t_mdp))
f.close()


