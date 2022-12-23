import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class RewardFunctionPR(torch.nn.Module):
    '''
    The partial return reward learning model
    '''
    def __init__(self,GAMMA, n_features=6):
        super(RewardFunctionPR, self).__init__()
        self.n_features = n_features
        self.GAMMA = GAMMA
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False)
        # torch.nn.init.normal_( self.linear1.weight, mean=0.0, std=0.3)
        # self.linear1.weight = torch.nn.Parameter(torch.ones_like(self.linear1.weight)*-100)


    def forward(self, phi):
        pr = torch.squeeze(self.linear1(phi))
        left_pred = torch.sigmoid(torch.subtract(pr[:,0:1],pr[:,1:2]))
        right_pred = torch.sigmoid(torch.subtract(pr[:,1:2],pr[:,0:1]))
        phi_logit = torch.stack([left_pred,right_pred],axis=1)
        return phi_logit

class RewardFunctionPRGen(torch.nn.Module):
    '''
    The partial return reward learning model, but generalizes the features rather than learning a weight for each individual feature.
    This is useful when the feature space is too large, such as when learning (state, action) pair values rather than reward feature values.
    '''
    def __init__(self,GAMMA, n_features=400):
        super(RewardFunctionPRGen, self).__init__()
        self.n_features = n_features
        self.GAMMA = GAMMA

        #input is (x,y,action_index)
        self.linear1 = torch.nn.Linear(3, 128)
        self.linear2 = torch.nn.Linear(128, 1)
        self.activation =  torch.nn.Tanh()

    def forward(self, sa_list):
        pr = torch.squeeze(self.linear2(self.activation(self.linear1(sa_list))))
        pr = torch.sum(pr, dim=2)

        left_pred = torch.sigmoid(torch.subtract(pr[:,0:1],pr[:,1:2]))
        right_pred = torch.sigmoid(torch.subtract(pr[:,1:2],pr[:,0:1]))
        phi_logit = torch.stack([left_pred,right_pred],axis=1)
        return phi_logit

    def get_trans_val(self,trans):
        return torch.squeeze(self.linear2(self.activation(self.linear1(trans))))


class RewardFunctionRegret(torch.nn.Module):
    '''
    The regret reward learning model
    '''
    def __init__(self,GAMMA,succ_feats,preference_weights, n_features=6,include_actions=False,succ_q_feats=None):
        super(RewardFunctionRegret, self).__init__()
        self.n_features = n_features
        self.GAMMA = GAMMA
        self.succ_feats = torch.tensor(succ_feats,dtype=torch.double).to(device)

        if succ_q_feats is not None:
            self.succ_q_feats = torch.tensor(succ_q_feats,dtype=torch.double).to(device)

        self.include_actions = include_actions
        # self.succ_feats_gt = torch.tensor(succ_feats_gt,dtype=torch.double)
        # self.w = torch.nn.Parameter(torch.tensor(np.zeros(n_features).T,dtype = torch.float,requires_grad=True))
        self.linear1 = torch.nn.Linear(self.n_features, 1,bias=False).double()

        self.softmax = torch.nn.Softmax(dim=1)

        #optionally set weights for the deterministic regret model
        if preference_weights is not None:
            self.rw = preference_weights[0][0]
            self.v_stw = preference_weights[0][1]
            self.v_s0w = preference_weights[0][2]
        else:
            self.rw = 1
            self.v_stw = 1
            self.v_s0w = 1

        self.T = 0.001

    def get_qs(self,state,action):
        '''
        Approximate Q* using successor feautres
        '''
        selected_succ_feats =[self.succ_q_feats[:,x,y,a].double() for (x,y),a in zip(state.long(),action.long())]
        selected_succ_feats = torch.stack(selected_succ_feats)

        qs = self.linear1(selected_succ_feats)
        q_pi_approx = torch.sum(torch.mul(self.softmax(qs/self.T),qs),dim = 1)

        q_pi_approx = torch.squeeze(q_pi_approx)
        return q_pi_approx

    def get_vals(self,cords):
        '''
        Approximate V* using successor feautres
        '''
        selected_succ_feats =[self.succ_feats[:,x,y].double() for x,y in cords.long()]
        selected_succ_feats = torch.stack(selected_succ_feats)
        vs = self.linear1(selected_succ_feats)
        v_pi_approx = torch.sum(torch.mul(self.softmax(vs/self.T),vs),dim = 1)

        v_pi_approx = torch.squeeze(v_pi_approx)
        return v_pi_approx

    def forward(self, phi):

        if self.include_actions:
            a1 = torch.squeeze(phi[:,:,6:7])
            a2 = torch.squeeze(phi[:,:,7:8])
            a3 = torch.squeeze(phi[:,:,8:9])

            s1_x = torch.squeeze(phi[:,:,9:10])
            s1_y = torch.squeeze(phi[:,:,10:11])
            s1 = torch.stack([s1_x,s1_y], dim=1)

            s2_x = torch.squeeze(phi[:,:,11:12])
            s2_y = torch.squeeze(phi[:,:,12:13])
            s2 = torch.stack([s2_x,s2_y], dim=1)

            s3_x = torch.squeeze(phi[:,:,13:14])
            s3_y = torch.squeeze(phi[:,:,14:15])
            s3 = torch.stack([s3_x,s3_y], dim=1)

            q1 = self.get_qs(s1,a1)
            v1 = self.get_vals(s1)
            adv1 = torch.subtract(v1,q1)
            left_adv1 = adv1[:,0:1]
            right_adv1 = adv1[:,1:2]

            q2 = self.get_qs(s2,a2)
            v2 = self.get_vals(s2)
            adv2 = torch.subtract(v2,q2)
            left_adv2 = adv2[:,0:1]
            right_adv2 = adv2[:,1:2]

            left_delta_er = torch.add(left_adv1, left_adv2)
            right_delta_er = torch.add(right_adv1, right_adv2)


            q3 = self.get_qs(s3,a3)
            v3 = self.get_vals(s3)
            adv3 = torch.subtract(v3,q3)
            left_adv3 = adv3[:,0:1]
            right_adv3 = adv3[:,1:2]

            left_delta_er = -torch.add(left_delta_er, left_adv3)
            right_delta_er = -torch.add(right_delta_er, right_adv3)
        else:
            pr = torch.squeeze(self.linear1(phi[:,:,0:self.n_features].double()))
            ss_x = torch.squeeze(phi[:,:,self.n_features:self.n_features+1])
            ss_y = torch.squeeze(phi[:,:,self.n_features+1:self.n_features+2])
            ss_cord_pairs = torch.stack([ss_x,ss_y], dim=1)


            es_x = torch.squeeze(phi[:,:,self.n_features+2:self.n_features+3])
            es_y = torch.squeeze(phi[:,:,self.n_features+3:self.n_features+4])
            es_cord_pairs = torch.stack([es_x,es_y], dim=1)


            #build list of succ fears for start/end states
            v_ss = self.get_vals(ss_cord_pairs)
            v_es = self.get_vals(es_cord_pairs)


            left_pr = pr[:,0:1]
            right_pr = pr[:,1:2]

            left_vf_ss = v_ss[:,0:1]
            right_vf_ss = v_ss[:,1:2]

            left_vf_es = v_es[:,0:1]
            right_vf_es = v_es[:,1:2]


            #apply weights learned from logistic regression (if it exists)
            left_pr = torch.multiply(left_pr, self.rw)
            right_pr = torch.multiply(right_pr, self.rw)

            left_vf_ss = torch.multiply(left_vf_ss, self.v_s0w)
            right_vf_ss = torch.multiply(right_vf_ss, self.v_s0w)

            left_vf_es = torch.multiply(left_vf_es, self.v_stw)
            right_vf_es = torch.multiply(right_vf_es, self.v_stw)

            #calculate change in expected return
            left_delta_v = torch.subtract(left_vf_es, left_vf_ss)
            right_delta_v = torch.subtract(right_vf_es, right_vf_ss)

            left_delta_er = torch.add(left_pr, left_delta_v)
            right_delta_er = torch.add(right_pr, right_delta_v)

        left_pred = torch.sigmoid(torch.subtract(left_delta_er, right_delta_er))
        right_pred = torch.sigmoid(torch.subtract(right_delta_er, left_delta_er))

        phi_logit = torch.stack([left_pred,right_pred],axis=1)

        return phi_logit
