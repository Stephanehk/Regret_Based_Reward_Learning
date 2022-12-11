import numpy as np
import json
import random


class GridWorldEnv:
    def __init__(self,board_name, height=10, width=10):


        self.prev_reward_function = None

        #Number of actions. We allow for the 4 cardinal directions.  
        self.action_space = 4 
        #Number of reward features. We use one feature for each component of the reward function, [gas, goal, sheep, coin, road block, mud]
        self.feature_size = 6
        #The ground truth reward vector, indicating the weights of each reward component.
        self.reward_array =[-1,50,-50,1,-1,-2]
        #A list of all possible actions (ie: the 4 cardinal directions)
        self.actions = [[-1,0],[1,0],[0,-1],[0,1]]

        self.ss = (0,0)
        self.pos = self.ss

        self.height = height
        self.width = width


        if board_name != None:
            self.n_starts = 92
            board_fp = "../MTURK_UI/assets/boards/" + board_name + "_board.json"
            reward_fp = "../MTURK_UI/assets/boards/" + board_name + "_rewards_function.json"
            #2021-07-29_sparseboard2-notrap_board.json
            with open(board_fp, 'r') as j:
                self.board = json.loads(j.read())

            with open(reward_fp, 'r') as j:
                self.reward_function = json.loads(j.read())
            self.generate_transition_probs()
        else:
            self.n_starts = 0
            self.board = np.zeros((height,width))
            self.reward_function = None
            # self.set_custom_reward_function(self.reward_array,set_global=True)

        self.observation_space = len(self.board)*len(self.board[0])
        self.transition_probs = None


    def generate_transition_probs(self):
        '''
        This function generates the transition dynamics of the MDP, which by default are deterministic. 
        '''
        probs = []
        for x in range (len(self.board)):
            width_nexts = []
            for y in range(len(self.board[0])):
                action_nexts = []
                for a_index in range(len(self.actions)):
                    next_state, reward, done, reward_feature = self.get_next_state((x,y), a_index)
                    action_nexts.append({(next_state, np.dot(reward_feature,self.reward_array), done, tuple(reward_feature)):1})
                width_nexts.append(action_nexts)
            probs.append(width_nexts)
        self.transition_probs = probs


    def set_start_state(self,ss):
        '''
        Sets the start state of the agent in the given MDP.

        Input
        - ss: a tuple storing the (x,y) coordinates of the desired start state.
        '''
        self.ss = ss
        self.pos = ss


    def find_n_starts(self):
        '''
        Finds the number of possible start states. This is the set of all non-terminal and non-blocking states. 
        '''
        self.n_starts = 0
        for x in range (len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] != 1 and self.board[x][y] != 2 and self.board[x][y] != 3 and self.board[x][y] != 7 and self.board[x][y] != 8 and self.board[x][y] != 8:
                    self.n_starts+=1

    def get_blocking_cords(self):
        '''
        Gets a list of blocking states (ie: states that are inaccesible by the agent)
        '''
        self.blocking_cords = []
        for x in range (len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] == 2 or self.board[x][y] == 8:
                    self.blocking_cords.append([x,y])

    def get_goal_rand(self):
        '''
        Gets a list of goal states. 
        '''
        self.goals = []
        for x in range (len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] == 1 or self.board[x][y] == 7:
                    self.goals.append((x,y))
        return random.choice(self.goals)


    def get_terminal_cords(self):
        '''
        Gets a list of terminal states. 
        '''
        self.terminal_cords = []
        for x in range (len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] == 1 or self.board[x][y] == 7 or self.board[x][y] == 3 or self.board[x][y] == 9:
                    self.terminal_cords.append([x,y])


    # def reset(self):
    #     self.pos = self.ss
    #     x,y = self.pos
    #     N = x + len(self.board[0])*y
    #     return N

    def set_custom_reward_function(self,reward_arr,set_global=True):
        '''
        Changes the ground truth reward function of the MDP.

        Input
        - reward_arr: a list containing the desired weights for each reward feature.
        - set_global: if true, sets MDPs reward function using reward_arr. If false, does nothing (useful for testing) 
        '''
        #[gas, goal, sheep, coin, roadblock, mud]
        reward_function = [[[0 for a in range (len(self.actions))] for x in range (len(self.board[0]))] for y in range(len(self.board))]
        for x in range (len(self.board)):
            for y in range(len(self.board[0])):
                for a_i in range(len(self.actions)):
                    a = self.actions[a_i]
                    state = [x,y]
                    next_state = [x+a[0],y+a[1]]

                    if self.board[x][y] == 3 or self.board[x][y] == 1 or self.board[x][y] == 7 or self.board[x][y] == 9:
                        #means current state is terminal
                        reward_function[x][y][a_i] = 0
                        continue

                    if next_state[0] < 0 or next_state[1] < 0 or next_state[0] >= self.height or next_state[1] >= self.width:
                        #invalid action
                        if self.board[state[0]][state[1]] < 6:
                            reward_function[x][y][a_i] = reward_arr[0]
                        else:
                            reward_function[x][y][a_i] = reward_arr[5]
                        continue

                    if self.board[next_state[0]][next_state[1]] == 0:
                        reward_function[x][y][a_i] = reward_arr[0] #blank
                    elif self.board[next_state[0]][next_state[1]] == 1:
                        reward_function[x][y][a_i] = reward_arr[1] #goal
                    elif self.board[next_state[0]][next_state[1]] == 2:
                        if self.board[x][y] < 6:
                            reward_function[x][y][a_i] = reward_arr[0] #blocking state
                        else:
                            reward_function[x][y][a_i] = reward_arr[5]
                    elif self.board[next_state[0]][next_state[1]] == 3:
                        reward_function[x][y][a_i] = reward_arr[2] #sheap
                    elif self.board[next_state[0]][next_state[1]] == 4:
                        reward_function[x][y][a_i] = reward_arr[3] + reward_arr[0] #coin
                    elif self.board[next_state[0]][next_state[1]] == 5:
                        reward_function[x][y][a_i] = reward_arr[4] + reward_arr[0] #roadblock
                    elif self.board[next_state[0]][next_state[1]] == 6:
                        reward_function[x][y][a_i] = reward_arr[5] #mud
                    elif self.board[next_state[0]][next_state[1]] == 7:
                        reward_function[x][y][a_i] = reward_arr[1] #goal
                    elif self.board[next_state[0]][next_state[1]] == 8:
                        if self.board[x][y] < 6:
                            reward_function[x][y][a_i] = reward_arr[0] #blocking state + mud
                        else:
                            reward_function[x][y][a_i] = reward_arr[5]
                    elif self.board[next_state[0]][next_state[1]] == 9:
                        reward_function[x][y][a_i] = reward_arr[2] #sheep
                    elif self.board[next_state[0]][next_state[1]] == 10:
                        reward_function[x][y][a_i] = reward_arr[3] + reward_arr[5] #coin + mud
                    elif self.board[next_state[0]][next_state[1]] == 11:
                        reward_function[x][y][a_i] = reward_arr[4] + reward_arr[5] #roadblock + mud
                    else:
                        print (self.board[next_state[0]][next_state[1]])
                        assert False
        if set_global:
            self.prev_reward_function = self.reward_function
            self.reward_function = reward_function
            self.reward_array = reward_arr
            self.generate_transition_probs()
        return reward_function



    def state2tab(self,x,y):
        '''
        Given coordinates in the MDP, converts them to a one-hot vector.
        '''
        N = x + len(self.board[0])*y
        ones = np.zeros(self.observation_space)
        ones[N] = 1
        return ones, N

    def is_blocked(self,x,y):
        '''
        Determines if the inputted coordinates are in a blocking state or not.
        '''
        if self.board[x][y] == 2 or self.board[x][y] == 8:
            return True
        else:
            return False

    def is_terminal(self,x,y):
        '''
        Determines if the inputted coordinates are in a terminal state or not.
        '''
        if self.board[x][y] == 3 or self.board[x][y] == 1 or self.board[x][y] == 7 or self.board[x][y] == 9:
            return True
        else:
            return False

    def is_goal(self,x,y):
        '''
        Determines if the inputted coordinates are a goal state or not.
        '''
        if self.board[x][y] == 1 or self.board[x][y] == 7:
            return True
        else:
            return False

    def is_valid_move(self,x,y,a):
        '''
        Given a set of coordinates and an action, determines if an action is valid. If an action attempts to move an agent outside the bounds of the board
        or into a blocking state it is invalid.

        Input:
        - x,y: the inputted coordinates.
        - a: the inputted action, represented as the array [x displacement, y displacement]

        Output:
        - true if the action is valid, false otherwise.
        '''
        if (x + a[0] >= 0 and x + a[0] < len(self.board) and y + a[1] >= 0 and y + a[1] < len(self.board[0])) and self.board[x + a[0]][y + a[1]] != 2 and self.board[x + a[0]][y + a[1]] != 8:
            return True
        else:
            return False


    def get_reward_feature(self,x,y,prev_x,prev_y):
        '''
        Returns the reward features for a given transition.

        Input:
        - x,y: the inputted coordinates.
        - prev_x,prev_y: the previous coordinates.
        
        Output:
        - A list of reward features for the given transition. 

        '''
        reward_feature = np.zeros(self.feature_size)
        if self.board[x][y] == 0:
            reward_feature[0] = 1
        elif self.board[x][y] == 1:
            #flag
            # reward_feature[0] = 1
            reward_feature[1] = 1
        elif self.board[x][y] == 2:
            #house
            # reward_feature[0] = 1
            pass
        elif self.board[x][y] == 3:
            #sheep
            # reward_feature[0] = 1
            reward_feature[2] = 1
        elif self.board[x][y] == 4:
            #coin
            # reward_feature[0] = 1
            reward_feature[0] = 1
            reward_feature[3] = 1
        elif self.board[x][y] == 5:
            #road block
            # reward_feature[0] = 1
            reward_feature[0] = 1
            reward_feature[4] = 1
        elif self.board[x][y] == 6:
            #mud area
            # reward_feature[0] = 1
            reward_feature[5] = 1
        elif self.board[x][y] == 7:
            #mud area + flag
            reward_feature[1] = 1
        elif self.board[x][y] == 8:
            #mud area + house
            pass
        elif self.board[x][y] == 9:
            #mud area + sheep
            reward_feature[2] = 1
        elif self.board[x][y] == 10:
            #mud area + coin
            # reward_feature[0] = 1
            reward_feature[5] = 1
            reward_feature[3] = 1
        elif self.board[x][y] == 11:
            #mud area + roadblock
            # reward_feature[0] = 1
            reward_feature[5] = 1
            reward_feature[4] = 1
        # else:
            #gas area
            # reward_feature[0] = 1
        if (x,y) == (prev_x, prev_y):
            reward_feature *= [1,0,0,0,0,1]

        return reward_feature

    def get_prev_state(self,s,a_index):
        '''
        Given a state and the previous action index, returns the previous state info.

        Input:
        - s: the inputted coordinates represented as the tuple (x,y)
        - a_index: the previous action index.
        
        Output:
        - prev_state: The previous state represented as the tuple (x,y). None if the previous transition is invalid. 
        - reward: The previously collected reward. None if the previous transition is invalid.
        - done: If the previous transition was into a terminal state or not. None if the previous transition is invalid. 
        - reward_feature: The reward feature for the previous transition. None if the previous transition is invalid.

        '''
        x,y = s
        done = False
        actions = [[-1,0],[1,0],[0,-1],[0,1]]
        a = actions[a_index]

        if self.board[x][y] == 3 or self.board[x][y] == 1 or self.board[x][y] == 7 or self.board[x][y] == 9:
            done = True

        prev_x = x-a[0]
        prev_y = y-a[1]

        if prev_x < 0 or prev_y < 0 or prev_x >= self.height or prev_y >= self.width:
            #means that the transition does not exist
            return None, None, None, None

        reward = self.reward_function[prev_x][prev_y][a_index]


        if self.is_valid_move(prev_x,prev_y,a):
            # x = x + a[0]
            # y = y + a[1]
            prev_state = (prev_x,prev_y)
            reward_feature = self.get_reward_feature(prev_x,prev_y)

        return prev_state, reward, done, reward_feature


    def get_next_state_prob(self,s,a_index):
        '''
        Given a state and the previous action index, returns the next state info for an MDP with stochastic transition dynamics.

        Input:
        - s: the inputted coordinates represented as the tuple (x,y)
        - a_index: the current action index.
        
        Output:
        - next_state: The next state represented as the tuple (x,y), sampled from the MDP next state distribution.  
        - reward: The collected reward.
        - done: If the current transition was into a terminal state or not. 
        - reward_feature: The reward feature for the current transition.

        '''

        x,y = s
        prev_x,prev_y = s
        done = False
        actions = [[-1,0],[1,0],[0,-1],[0,1]]
        a = actions[a_index]

        if self.board[x][y] == 3 or self.board[x][y] == 1 or self.board[x][y] == 7 or self.board[x][y] == 9:
            done = True

        # reward = self.reward_function[x][y][a_index]

        transitions= self.transition_probs[x][y][a_index]
        trans = random.choices(list(transitions.keys()), weights=transitions.values(), k=1)
        next_state, reward, done, phi = trans[0]

        if len(self.reward_array) > 6:
            #means we are using extended SF
            action_phi = np.zeros((self.height,self.width,4))
            action_phi[x][y][a_index] = 1
            action_phi = np.ravel(action_phi)
            reward += np.dot(action_phi,self.reward_array[6:])

     
        return next_state, reward, done, list(phi)

    def get_next_state(self,s,a_index):
        '''
        Given a state and the previous action index, returns the next state info for an MDP with deterministic transition dynamics.

        Input:
        - s: the inputted coordinates represented as the tuple (x,y).
        - a_index: the current action index.
        
        Output:
        - next_state: The next state represented as the tuple (x,y).
        - reward: The collected reward.
        - done: If the current transition was into a terminal state or not. 
        - reward_feature: The reward feature for the current transition.

        '''

        x,y = s
        prev_x,prev_y = s
        done = False
        actions = [[-1,0],[1,0],[0,-1],[0,1]]
        a = actions[a_index]

        if self.board[x][y] == 3 or self.board[x][y] == 1 or self.board[x][y] == 7 or self.board[x][y] == 9:
            done = True

        reward = self.reward_function[x][y][a_index]

        if len(self.reward_array) > 6:
            #means we are using extended SF
            action_phi = np.zeros((self.height,self.width,4))
            action_phi[x][y][a_index] = 1
            action_phi = np.ravel(action_phi)
            reward += np.dot(action_phi,self.reward_array[6:])

        if self.is_valid_move(x,y,a) and not self.is_terminal(x,y):
            x = x + a[0]
            y = y + a[1]
        next_state = (x,y)


        reward_feature = self.get_reward_feature(x,y,prev_x,prev_y)

        return next_state, reward, done, reward_feature

    def step(self,a_index):
        '''
        Given an action index, returns the next state info and stores next state as a class variable. This is for deterministic transitions.

        Input:
        - a_index: the current action index.
        
        Output:
        - next_state: The next state represented as the tuple (x,y).
        - reward: The collected reward.
        - done: If the current transition was into a terminal state or not. 
        - reward_feature: The reward feature for the current transition.

        '''
        next_state, reward, done, reward_feature = self.get_next_state(self.pos, a_index)
        self.pos = next_state
        return next_state, reward, done, reward_feature

    def step_prob(self,a_index):
        '''
        Given an action index, returns the next state info and stores next state as a class variable. This is for stochastic transitions.

        Input:
        - a_index: the current action index.
        
        Output:
        - next_state: The next state represented as the tuple (x,y), sampled from the next state distribution.
        - reward: The collected reward.
        - done: If the current transition was into a terminal state or not. 
        - reward_feature: The reward feature for the current transition.

        '''
        next_state, reward, done, reward_feature = self.get_next_state_prob(self.pos, a_index)
        self.pos = next_state
        return next_state, reward, done, reward_feature

    def reset(self):
        '''
        Resets all class variables. This should be called at the end of every episode. 
        '''

        x = random.randrange(0,self.height)
        y = random.randrange(0,self.width)
        while self.is_blocked(x,y) or self.is_terminal(x,y):
            x = random.randrange(0,self.height)
            y = random.randrange(0,self.width)

        self.ss = (x,y)
        self.pos = self.ss
        return self.pos

    def find_action_index(self, action):
        '''
        Finds the index of an action represented as an array.

        Input:
        - The specified action, represented as the array [x displacement, y displacement]

        Output:
        - The action index, or false if the action does not exist. 
        '''

        actions = [[-1,0],[1,0],[0,-1],[0,1]]
        i = 0
        for a in actions:
            if a[0] == action[0] and a[1] == action[1]:
                return i
            i+=1
        return False


# env = GridWorldEnv("2021-07-29_sparseboard2-notrap")
