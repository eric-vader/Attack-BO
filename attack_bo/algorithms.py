#!/usr/bin/env python3
import common
import logging
import numpy as np
import scipy
from scipy import optimize
from datasets import TargetedAttack

import sys
CURRENT_MODULE = sys.modules[__name__]

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO

class NoStrategy(object):
    def __init__(self, **kwargs):
        pass
    def optimize_var(self, var, is_in_region):
        return var

class BaseStrategy(NoStrategy):
    def __init__(self, **kwargs):
        self.success_log = []
    def optimize_var(self, var, is_in_region):
        # We log the success rate first
        if is_in_region: 
            self.success_log.append(1)
        else:
            self.success_log.append(0)
        return var

class SimpleStrategy(BaseStrategy):
    def __init__(self, factor, consecutive_successes, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.consecutive_successes = consecutive_successes
    def optimize_var(self, var, is_in_region):
        var = super().optimize_var(var, is_in_region)

        # We now optimize
        n_logs = len(self.success_log)
        if n_logs > self.consecutive_successes and all(self.success_log[-self.consecutive_successes:]):
            return var - self.factor * var
        else:
            return var + self.factor * var

class NoAttack(common.Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)
    # The argument is the observable states
    def perform_attack(selected_x, budget=None,reward=None):
        return 0
    def get_additional_plots(self):
        return []

# Clipping would only make sense with region
class ClippingAttack(common.Component):
    def __init__(self, delta, strategy_cls="NoStrategy", strategy_kwargs={}, f_x_tilda=None, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)
        self.delta = delta
        self.f_x_tilda = f_x_tilda
        StrategyCls = getattr(CURRENT_MODULE, strategy_cls)
        self.opt_strategy = StrategyCls(**strategy_kwargs)
    def prepare(self, datasets, **kwargs):

        assert(isinstance(datasets, TargetedAttack))
        wrapped_fn = datasets.wrapped_fn
        self.f = wrapped_fn.obj_fn
        self.region = datasets.region

        if self.f_x_tilda == None:
            self.f_x_tilda, f_x_tilda_opt = wrapped_fn.find_fmin(self.f, datasets.region.bnds, 0.5, 200)
            logging.info(f"Auto inferred - f_x_tilda={self.f_x_tilda}, f_x_tilda_opt={f_x_tilda_opt}")

    # The argument is the observable states
    def perform_attack(self, selected_x, budget=None,reward=None):
        self.delta = self.opt_strategy.optimize_var(self.delta, self.region.is_within(selected_x))
        return self.perturb_h(selected_x)
    def perturb_h(self, x):        
        if self.region.is_within(x):
            return 0
        # delta is positive, considering fmin, it is + delta instead of - delta
        return np.maximum(0.0, -self.f(x) + self.f_x_tilda + self.delta )
    def get_additional_plots(self):
        return [ (lambda x: self.f(x) + self.perturb_h(x), 'b--', 'Clipped $f$') ]

class SubtractionAttack(common.Component):
    def __init__(self, bump_fn, x_scale, y_scale, x_shift, y_shift, **kwargs):
        super().__init__(**kwargs)

        if type(bump_fn) == str:
            self.bump_fn =  eval(bump_fn)
        else:
            assert(False)
            
        self.x_scale = x_scale
        self.y_scale = y_scale

        self.x_shift = x_shift
        self.y_shift = y_shift

    def prepare(self, datasets, **kwargs):
        self.f = datasets.wrapped_fn.obj_fn
        self.region = datasets.region

        '''
        print(datasets.wrapped_fn.fmin_optima)
        print(datasets.wrapped_fn.bnds)
        print(self.f)
        bnds = np.array(datasets.wrapped_fn.bnds[0], dtype=np.float64)
        bnds1 = np.array(bnds)
        bnds1[1] = datasets.wrapped_fn.fmin_optima[0][0]
        print(bnds1)
        bnds2 = np.array(bnds)
        bnds2[0] = datasets.wrapped_fn.fmin_optima[0][0]
        
        print(bnds2)
        sol1 = optimize.least_squares(self.f, datasets.wrapped_fn.fmin_optima[0], bounds=bnds1)
        sol2 = optimize.least_squares(self.f, datasets.wrapped_fn.fmin_optima[0], bounds=bnds2)
        print(sol1)
        print(sol2)
        '''
    # The argument is the observable states
    def perform_attack(self, selected_x, budget=None,reward=None):
        return -(self.y_scale*self.bump_fn((selected_x-self.x_shift)/self.x_scale) + self.y_shift)

    def get_additional_plots(self):
        return [ (lambda x: self.f(x)+self.perform_attack(x), 'b--', '$f(x)-h(x)$'),
                 (lambda x: self.perform_attack(x), 'r--', '$-h(x)$')
               ]

class AggressiveSubtractionAttack(common.Component):
    def __init__(self, delta=None, strategy_cls="NoStrategy", strategy_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)
        self.delta = delta
        StrategyCls = getattr(CURRENT_MODULE, strategy_cls)
        self.opt_strategy = StrategyCls(**strategy_kwargs)
    def prepare(self, datasets, **kwargs):
        self.f = datasets.wrapped_fn.obj_fn
        self.region = datasets.region
        wrapped_fn = datasets.wrapped_fn

        if self.delta == None:
            f_x_tilda, f_x_tilda_opt = wrapped_fn.find_fmin(self.f, datasets.region.bnds, 0.5, 200)
            self.delta = f_x_tilda - wrapped_fn.fmin
            logging.info(f"Auto inferred - delta={self.delta}, f_x_tilda={f_x_tilda}, f_x_tilda_opt={f_x_tilda_opt}")

    # The argument is the observable states
    def perform_attack(self, selected_x, budget=None,reward=None):
        self.delta = self.opt_strategy.optimize_var(self.delta, self.region.is_within(selected_x))
        return self.neg_h(selected_x)
    def neg_h(self, x):        
        if self.region.is_within(x): 
            return 0
        return self.delta
    def get_additional_plots(self):
        return [ (lambda x: self.f(x)+self.neg_h(x), 'b--', '$f(x)-h(x)$') ]

class RandomAttack(common.Component):
    def __init__(self, mean, var, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)
        self.mean = mean
        self.sd = np.sqrt(var)
    # The argument is the observable states
    def perform_attack(self, selected_x, budget=None,reward=None):
        return self.random_state.normal(self.mean, self.sd)
    def get_additional_plots(self):
        return []

class PPOAttack(common.Component):
    def __init__(self, total_timesteps, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)
        self.total_timesteps = total_timesteps
    def prepare(self, datasets, **kwargs):

        # This will be cached; we use the type_hash so that the model is captured regardless of the random seed
        # use hash for algorithm as it would change the model; the dataset starting point does not matter for training
        model_filename = f'A{self.hash}-D{datasets.type_hash}.model'
        model_path = self.config.configs['cache'].get_path(model_filename)
        
        logging.info(f'PPOAttack is attempting to load {model_path}')

        if self.config.configs['cache'].does_exist(model_filename):
            logging.info(f'PPOAttack model loaded from {model_path}')
            # Load the model if exist
            self.model = PPO.load(model_path)
        else:
            logging.info(f'PPOAttack model is being trained.')
            # Train and save the model
            datasets.is_training = True
            env = DummyVecEnv([lambda: datasets]) 

            self.model = PPO(MlpPolicy, env, verbose=1, seed=self.random_seed)
            self.model.learn(total_timesteps=self.total_timesteps)
            datasets.is_training = False
            datasets.reset()
            logging.info(f'PPOAttack model saved to {model_path}')
            self.model.save(model_path)


    # The argument is the observable states
    def perform_attack(self, selected_x, budget=None, reward=None):
        action, _states = self.model.predict(selected_x)
        return float(action)
    def get_additional_plots(self):
        return []
