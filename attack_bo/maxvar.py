from GPyOpt.acquisitions import AcquisitionBase

"""
MaxVar + Elimination: 
At each time, define the set of potential minimizers is to contain 
all points whose LCB is at least as low as the lowest UCB. 
Then, select the point with the highest posterior variance.
"""
class AcquisitionMaxVar(AcquisitionBase):

    analytical_gradient_prediction = False

    # Fake init
    def __init__(self):
        pass

    def delayed_init(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(AcquisitionMaxVar, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  

        self.dict_lowest_ucb = {}
    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        """
        n_samples = self.model.model.X.shape[0]
        # Once every model update
        if not n_samples in self.dict_lowest_ucb:
            # Here we calculate the lowest UCB point
            lowest_ucb = self.optimizer.optimize(f=self.ucb_f, f_df=self.ucb_f_df)
            l_ucb_x, l_ucb_v = lowest_ucb
            self.dict_lowest_ucb[n_samples] = l_ucb_v
        else:
            # Retrieve the prev computed lowest UCB point for this model version
            l_ucb_v = self.dict_lowest_ucb[n_samples]

        m, s = self.model.predict(x)

        # LCB
        lcb = m - self.exploration_weight * s

        # Set to zero
        s[lcb > l_ucb_v] = 0.0
                
        return s

    def _compute_acq_withGradients(self, x):
        raise NotImplementedError

    def ucb_f(self, x):
        m, s = self.model.predict(x)   
        f_acqu = m + self.exploration_weight * s
        return f_acqu

    def ucb_f_df(self, x):
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        f_acqu = m + self.exploration_weight * s       
        df_acqu = dmdx + self.exploration_weight * dsdx
        return f_acqu, df_acqu