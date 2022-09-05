import numpy as np
import matplotlib.pyplot as plt

models = ["er_er", "pr_pr"]
pref_types = ["sigmoid", "deterministic"]
discounts = np.array([0.5, 0.9, 0.99, 0.999])
all_res = {}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

for model in models:
    for pref_type in pref_types:
        discount_res = []
        for discount in discounts:
            fp = "../discounting_expirements/100_200_num_prefs=3000main_avg_return_" + pref_type + "_" + model + "False_" + str(discount) + ".npy"
            res = np.load(fp)

        
            discount_res.append(sum(i > 0.9 for i in res)/len(res))
            # print (len(res))
            # if len(res) != 100:
            print (model, pref_type, discount)
            print (res)
            print (sum(i > 0.9 for i in res)/len(res))


        if model == "er_er":
            color = "blue"
        elif model == "pr_pr":
            color = "red"
        elif model == "er_pr":
            color = "purple"
        elif model == "pr_er":
            color = "pink"

        if pref_type == "sigmoid":
            plt.plot(discounts.astype('str'),discount_res,'--',color=color)
        else:
            plt.plot(discounts.astype('str'),discount_res,color=color)

# plt.yticks([-9,-7,-5,-3,-1,0,1])
plt.xticks(discounts.astype('str'))
plt.ylim(0,1.2)
plt.xlim(discounts.astype('str')[0],discounts.astype('str')[-1])


plt.hlines(y=1, xmin=discounts.astype('str')[0], xmax=discounts.astype('str')[-1], colors='grey', linestyles='--', lw=1, label='Single Short Line')

plt.savefig("test.png", dpi=300)
            




