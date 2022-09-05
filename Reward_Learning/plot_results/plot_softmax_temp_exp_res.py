import numpy as np
import matplotlib.pyplot as plt

#MDP 1
# regret_y = [1,1,1,1,0.867,0.6,0.367,0.3]
regret_y = [1,1,1,1,0.967,0.8,0.533,0.467]

# pr_y = [1,1,1,0.8,0.4,0.267,0.167,0.033]
pr_y = [1,1,1,0.967,0.8,0.567,0.433,0.4]

#MDP 2
# regret_y_2 = [1,1,1,1,0.967,0.767,0.467,0.4]
regret_y_2 = [1,1,1,1,0.967,0.8,0.533,0.467]

# pr_y_2 = [1,1,1,1,0.833,0.467,0.267,0.233]
pr_y_2 = [1,1,1,1,0.867,0.733,0.5,0.5]

x = np.array([0.1,1,2,5,10,20,50,100])

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.plot(x.astype('str'),regret_y,color="blue")
plt.plot(x.astype('str'),pr_y,color="red",alpha=0.7)

plt.plot(x.astype('str'),regret_y_2,color="blue",linestyle="dashdot")
plt.plot(x.astype('str'),pr_y_2,color="red",alpha=0.7,linestyle="dashdot")

# plt.yticks([-9,-7,-5,-3,-1,0,1])
plt.xticks(x.astype('str'))
plt.ylim(0,1.2)
plt.xlim(x.astype('str')[0],x.astype('str')[-1])


plt.hlines(y=1, xmin=x.astype('str')[0], xmax=x.astype('str')[-1], colors='grey', linestyles='--', lw=1, label='Single Short Line')

plt.savefig("test_2.png", dpi=300)
            




