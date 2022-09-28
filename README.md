# Regret Based Reward Learning
This is the official implementation for the paper [Models of human preference for learning reward functions](https://arxiv.org/pdf/2206.02231.pdf).

### MTURK UI
This folder contains the scripts necessary to run our MTURK human expirements. All assets, included the segments shown to users, are included. You will need to add your MTURK credentials to ```startMTurk.py```. To run the expirement, start the MTURK HIT using startMTurk.py and the web socket by running ```workerHandlerSocket.py```. A video showing the full experimental protocol can be seen [here](https://www.youtube.com/watch?v=zpIkVAHRm1Y).

### Segment Selection
This folder contains various board configurations used during the MTURK expirements, as well as the script ```generate_segment_pairs.py``` which finds all segments in our segment space for a given board confiugarion, and organizes them into categories from the first stage of data collection (see appendix D.3).

### Dataset
Our dataset of human preferences can be found in the MTURK_Data folder. Folders are formatted as DATE_questions.data, which contain information about the segments shown to each subject, and DATA_answers.data, which contain the subjects preferences. The folders entitled DATE_data_samples contain the segment images shown to each user. Please see ```Reward_Learning/load_training_data.py``` for various functions used to load and preprocess this data.

### Reward Learning
This folder contains the core expirements of our paper.

#### Descriptive results
Running the script ```logistic_reg.py``` will generate the logistic regression results in table 1 of the paper. These results display the likelihood of human preferences under different preference models using our dataset of human preferences.

#### Results from learning reward functions
There are several options for learning a reward function for the original delivery domain. Below we list a few:

* Using the regret reward learning model with stochastically generated synthetic preferences:
  * ```python3 reward_learning.py --preference_assum regret --preference_model regret --N_ITERS 5000 --LR 2 --mode sigmoid```
* Using the regret reward learning model with deterministically generated synthetic preferences:
  * ```python3 reward_learning.py --preference_assum regret --preference_model regret --N_ITERS 5000 --LR 2 --mode deterministic```
* Using the regret reward learning model with human preferences:
  * ```python3 reward_learning.py --preference_assum regret --preference_model regret --N_ITERS 5000 --LR 2 --mode deterministic_human_preferences```
* Using the partial return reward learning model with stochastically generated synthetic preferences:
  * ```python3 reward_learning.py --preference_assum pr --preference_model pr --N_ITERS 30000 --LR 0.5 --mode sigmoid```
* Using the partial return reward learning model with deterministically generated synthetic preferences:
  * ```python3 reward_learning.py --preference_assum pr --preference_model pr --N_ITERS 30000 --LR 0.5 --mode deterministic```
* Using the partial return reward learning model with human preferences:
  * ```python3 reward_learning.py --preference_assum pr --preference_model pr --N_ITERS 30000 --LR 0.5 --mode deterministic_human_preferences```

To run the reward learning expirements on a set of randomly generated MDPs, specify the flag ```--use_random_mdps```. Note that you will have to generate these random MDPs by running the script ```generate_random_mdp.py```. A complete list of all the arguments for reward learning can be found by running ```python3 reward_learning.py -h```
