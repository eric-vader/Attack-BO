#!/usr/bin/env python3
import common
import logging

import plotly.graph_objects as go
import pylab as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from GPyOpt.models.gpmodel import GPModel
import numpy as np
import GPy
import gym
from gym import spaces
from GPyOpt.core.task.space import Design_space
from gpy_wrapper import GpyWrapper
import random
import math
import imageio
from inspect import signature
from collections import defaultdict
import pickle

from GPy import kern

from hpolib.benchmarks import synthetic_functions
# Hacks to get the import to work.
from hpolib.benchmarks.synthetic_functions.rosenbrock import Rosenbrock5D
synthetic_functions.Rosenbrock5D = Rosenbrock5D

from skopt import sampler
from skopt.space import Space
from scipy.spatial.distance import pdist

import os
import robot_pushing

class BayesOptEnv(gym.Env, common.Component):
    metadata = {'render.modes': ['live', 'file', 'none']}
    # domain: The input of the objective function

    def __init__(self, n_iter, initial_design_numdata, render_kwargs, fn_cls, fn_params, bo_model_params, reward_fn, **kwargs):
        super().__init__(**kwargs)

        # Now create the wrapped fn
        Fn = globals()[fn_cls]
        self.wrapped_fn = Fn(random_state=self.random_state, **fn_params)

        domain = self.wrapped_fn.gpy_domain()

        # The adversarial noise it can accept is 1d, always the same shape as the range of the fx
        # TODO we assume 1D for now
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float)
        
        # This is the point followed by the evaluated value
        # ie. (x1, x2, ..., y)
        space = Design_space(domain)
        low, high = map(list, zip(*space.get_bounds()))
        low, high = np.array(low), np.array(high)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float)
        
        self.img = None
        
        self.n_iter = n_iter
        self.initial_design_numdata = initial_design_numdata
        self.render_kwargs = render_kwargs
        self.reward_fn = reward_fn 
        self.is_training = False

        # Find the fmin and fmin_opt
        self.mlflow_logger.log_metrics({
            'fmin':self.wrapped_fn.fmin,
            'fmax':self.wrapped_fn.fmax,
            'ptp':self.wrapped_fn.ptp}, None)

        # Initalize the bo model
        self.bo_model_params = bo_model_params
        self.x0 = self.reset()

        self.metrics_log_fns = []

        # This is so we can viz the fn later
        if render_kwargs['mode'] == 'viz':
            
            N_samples = render_kwargs['N_samples']
            del render_kwargs['N_samples']

            chart_ref = render_kwargs['chart_ref']
            del render_kwargs['chart_ref']

            # self.wrapped_fn.num_dim
            grid_size = int(N_samples**(1./self.wrapped_fn.num_dim))
            x_grids = [ np.linspace(ax_bnds[0], ax_bnds[1], grid_size) for ax_bnds in zip(low,high) ]
            X = np.hstack([ x_i.reshape(-1, 1) for x_i in np.meshgrid(*x_grids) ])

            #Y = np.array([ self.wrapped_fn.obj_fn(_x) for _x in X ])
            Y = np.apply_along_axis(self.wrapped_fn.obj_fn, 1, X).reshape(tuple( grid_size for i in range(self.wrapped_fn.num_dim) ))

            with open(self.config.configs['data'].get_path(f'{chart_ref}.pkl'), 'wb') as pkl_file:
                #print(os.path.realpath(pkl_file.name))
                plot_data = {
                    'x_grids': x_grids,
                    'X': X,
                    'Y': Y,
                    'fmin':self.wrapped_fn.fmin,
                    'fmin_optima':self.wrapped_fn.fmin_optima,
                    'region': {
                        'min_coord': self.region.min_coord,
                        'max_coord': self.region.max_coord}
                }
                #print(plot_data)
                pickle.dump(plot_data, pkl_file)

        # Only for debugging and only for 1D
        if self.wrapped_fn.num_dim == 1 and render_kwargs['mode'] == 'live':
            import matplotlib.pyplot as plt
            lo,hi = low, high
            # 100 linearly spaced numbers
            x = np.linspace(lo,hi,10000).reshape(-1,1)

            # the function, which is y = x^2 here
            y = np.array([ self.wrapped_fn.obj_fn(_x) for _x in x ])

            # setting the axes at the centre
            fig = plt.figure(figsize=[16, 12], dpi=100)
            ax = fig.add_subplot(1, 1, 1)
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            # plot the function
            from scipy.signal import find_peaks
            id_max,_ = find_peaks(-y)
            minima_x = x[id_max]
            minima_y = y[id_max]

            plt.scatter(minima_x, minima_y)
            plt.plot(x,y, 'r')

            for m_x, m_y in zip(minima_x,minima_y):

                label = f"({m_x[0]:.4f},{m_y:.4f})"
                # print(f"({m_x[0]:.5f},{m_y:.5f})")

                plt.annotate(label, # this is the text
                            (m_x,m_y), # this is the point to label
                            textcoords="offset points", # how to position the text
                            xytext=(0,10), # distance from text to points (x,y)
                            ha='center') # horizontal alignment can be left, right or center

            # show the plot
            plt.savefig(self.config.configs['data'].get_path('fn.png'))

            plt.show()
    def reset(self):
        self.t = 0
        self.metrics_dict = {}
        self.metrics_dict['fx_best'] = math.inf
        self.metrics_dict['cum_instant_regret'] = 0
        self.metrics_dict['cum_best_regret'] = 0
        self.metrics_dict['cum_c_t'] = 0

        self.bo = self.create_bo_model(wrapped_fn=self.wrapped_fn, initial_design_numdata=self.initial_design_numdata, **self.bo_model_params)
        self.x0 = self.bo.suggested_sample

        return self.x0

    def step(self, action):

        # quick fix for PPO
        # PPO trains by [action] and tests by action
        # TODO should change this to [action] for everything.
        action = float(action)

        # Should not reach here ever.
        if self.t > self.n_iter + self.initial_design_numdata:
            logging.error("Agent should stop on Terminate.")
            return None, 0, True, {}
        
        next_eval = self.bo.opt_step(action)
        
        # distance between fmin and the current f value
        fx = self.wrapped_fn.metrics_history['f_x'][-1][0]
        if 'z_t' in self.wrapped_fn.metrics_history:
            z = self.wrapped_fn.metrics_history['z_t'][-1][0]
        else:
            z = 0
        self.metrics_dict['fx_best'] = min(fx, self.metrics_dict['fx_best'])
        reward = self.reward_fn(next_eval)
        terminate = False

        instant_regret = fx - self.wrapped_fn.fmin
        self.metrics_dict['cum_instant_regret'] += instant_regret
        best_regret = self.metrics_dict['fx_best'] - self.wrapped_fn.fmin
        self.metrics_dict['cum_best_regret'] += best_regret
        self.metrics_dict['cum_c_t'] += abs(action)

        #print(next_eval, self.fmin_optima)
        metrics = {
            't' : self.t,
            'y' : (fx + z + action),
            'fx' : fx,
            'z_t' : z,
            'c_t' : action,
            'cum_c_t' : self.metrics_dict['cum_c_t'],
            'norm_cum_c_t' : self.metrics_dict['cum_c_t']/self.wrapped_fn.ptp,
            'reward' : reward,
            'instant_regret' : instant_regret,
            'best_regret' : best_regret,
            'cum_instant_regret' : self.metrics_dict['cum_instant_regret'],
            'cum_best_regret' : self.metrics_dict['cum_best_regret'],
            'avg_cum_instant_regret' : self.metrics_dict['cum_instant_regret']/(self.t+1),
            'avg_cum_best_regret' : self.metrics_dict['cum_best_regret']/(self.t+1)
        }

        # Real datasets may not have fmin_opt
        if not self.wrapped_fn.fmin_optima is None:
            metrics['dist_fopt'] = np.linalg.norm(next_eval-self.wrapped_fn.fmin_optima)

        # Update with additional metrics
        for metrics_log_fn in self.metrics_log_fns:
            metrics = {**metrics, **metrics_log_fn(next_eval, **metrics)}

        if not self.is_training:
            self.mlflow_logger.log_metrics(metrics, self.t)
            # print(metrics, self.t)

        # Check if this is the last
        self.t += 1
        if self.t == self.n_iter + self.initial_design_numdata:
            terminate = True

        return next_eval, reward, terminate, {}
    def render(self, figsize=None, mode='live', opt='min', close=False, fps=1):

        if mode == 'none':
            return

        if self.t == 1:
            self.fig = plt.figure(1,figsize=figsize) # 
            if mode != 'live':
                matplotlib.use("Agg")
            else:
                plt.show(block=False)

        # plot 
        if self.t >= self.initial_design_numdata:
            run_i = self.t - self.initial_design_numdata
            current_png_path = self.config.configs['run'].get_path(f'{run_i:03d}.png')
            self.bo.plot_acquisition_noblock(self.fig, current_png_path, opt)

        # plot the animation at the end
        if self.t == self.n_iter + self.initial_design_numdata:
            with imageio.get_writer(self.config.configs['run'].get_path('animation.gif'), mode='I', fps=fps) as writer:
                for run_i in range(self.n_iter+1):
                    png_path = self.config.configs['run'].get_path(f'{run_i:03d}.png')
                    writer.append_data(imageio.imread(png_path))
                    os.remove(png_path)

    def create_bo_model(self, wrapped_fn, initial_design_numdata, kernel_cls, model_params, bo_params, kernel_param, prefit_param=None):

        Kernel = getattr(kern, kernel_cls)
        matern_kernel = Kernel(input_dim=wrapped_fn.num_dim, **kernel_param)

        model = GPModel(kernel=matern_kernel, **model_params)

        # Pre-fit the Matern Kernel to find some variance and lengthscale
        if prefit_param != None:
            self.prefit_kernel(matern_kernel, model, wrapped_fn.bnds, **prefit_param)
       
        return GpyWrapper(initial_design_numdata=initial_design_numdata, verbosity=False, 
            verbosity_model=False, model=model, f=wrapped_fn, domain=wrapped_fn.gpy_domain(), 
            **bo_params)

    def prefit_kernel(self, matern_kernel, model, bnds, sampler_cls, prefit_n, fixed_kern):

        # Old mesh code, simply just divide equally
        # mesh_lin = []
        # for d_bnd in bnds:
        #     l, h = d_bnd
        #     mesh_lin.append(np.linspace(l, h, prefit_n))
        # mesh_lin = np.meshgrid(*mesh_lin)
        # samples_x = np.array([ea_lin.flatten() for ea_lin in mesh_lin]).T

        # https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method-integer.html
        space = Space(np.array(bnds, dtype=np.float64))

        Sampler = getattr(sampler, sampler_cls)
        # We sample 10 * prefit_n and discard some, due to bug in Sampler.
        # In case that is not sufficient, we supplement using random samples.
        # See https://github.com/scikit-optimize/scikit-optimize/issues/1036
        try:
            samples_x = np.array(Sampler().generate(space.dimensions, prefit_n*10, random_state=self.random_state))[:prefit_n]
        except MemoryError as e:
            logging.error(e)
            logging.error("Memory overflow!")
            samples_x = np.array(Sampler().generate(space.dimensions, 500, random_state=self.random_state))[:prefit_n]

        if len(samples_x) < prefit_n:
            logging.error("Filling in random points for prefitting due to bug in Sampler.")
            # TODO Hack, The rest is filled by random, due to bug in Sampler
            samples_x = np.append(samples_x, space.rvs(prefit_n - len(samples_x), random_state=self.random_state), axis=0)

        X_init, Y_init = samples_x, np.apply_along_axis(self.wrapped_fn.obj_fn, 1, samples_x).reshape(-1, 1)

        model._create_model(X_init, Y_init)
        model.model.Gaussian_noise.fix()
        model.model.optimize_restarts(num_restarts=10, optimizer=model.optimizer, max_iters = 1000)
        model.model.Gaussian_noise.unfix()
        logging.info(f"Matern Kernel updated to var-{matern_kernel.variance.values}, ls-{matern_kernel.lengthscale.values}" )

        if fixed_kern:
            model.model.Gaussian_noise.fix()
            matern_kernel.lengthscale.fix()
            matern_kernel.variance.fix()
    def create_fn(self, obj_fn, random_state, **fn_params):
        raise NotImplemented

class Function(object):
    def __init__(self, obj_fn, bnds, random_state, scale=1.0):
        self.scale = scale
        if type(obj_fn) == str:
            _obj_fn = eval(obj_fn)
        else:
            _obj_fn = obj_fn
        self.obj_fn = lambda x: _obj_fn(x) * self.scale

        self.num_dim = len(bnds)
        self.random_state = random_state
        self.bnds = bnds
        self.metrics_history = defaultdict(list)
        self.init_fmin()
    def find_fmin(self, obj_fn, bnds, stepsize, niter):
        
        # Adapted from 
        # https://stackoverflow.com/questions/21670080/how-to-find-global-minimum-in-python-optimization-with-bounds

        basinhopping_x0 = []
        xmin = []
        xmax = []
        for bnd in bnds:
            basinhopping_x0.append(np.mean(bnd))
            xmin.append(bnd[0])
            xmax.append(bnd[1])

        class RandomDisplacementBounds(object):
            """random displacement with bounds"""
            def __init__(self, xmin, xmax, stepsize=0.5):
                self.xmin = xmin
                self.xmax = xmax
                self.stepsize = stepsize

            def __call__(self, x):
                """take a random step but ensure the new position is within the bounds"""
                while True:
                    # this could be done in a much more clever way, but it will work for example purposes
                    xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
                    if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                        break
                return xnew

        take_step = RandomDisplacementBounds(xmin, xmax, stepsize=stepsize)
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bnds)
        res = scipy.optimize.basinhopping(obj_fn, basinhopping_x0, niter=100, take_step=take_step, 
            minimizer_kwargs=minimizer_kwargs)

        return obj_fn(res.x), np.array([res.x])
    def gpy_domain(self):
        domain = []
        for d, d_bnd in enumerate(self.bnds):
            domain.append({'name': f'x_{d}', 'type': 'continuous', 'domain': d_bnd})
        return domain
    def __call__(self, X):
        F_X = np.apply_along_axis(self.obj_fn, 1, X).reshape(-1, 1) * self.scale
        self.metrics_history['f_x'].extend(F_X)
        return F_X
    def init_fmin(self):
        raise NotImplementedError

class NoisyFunction(Function):
    def __init__(self, noise_mean, noise_var, **kwargs):
        super().__init__(**kwargs)
        self.noise_mean = noise_mean
        self.noise_sd = np.sqrt(noise_var)
        
    def __call__(self, X):
        fx =  super().__call__(X)

        Z = self.random_state.normal(self.noise_mean, self.noise_sd, len(X)).reshape(-1, 1)
        self.metrics_history['z_t'].extend(Z)

        return fx + Z
    def init_fmin(self):
        # TODO We can consider cacheing this
        self.fmin, self.fmin_optima = self.find_fmin(self.obj_fn, self.bnds, stepsize=0.5, niter=100*len(self.bnds))
        logging.info(f"fmin=f({self.fmin_optima})={self.fmin} found by BasinHopping. Please double check value.")

        # This is used to find fmax to find ptp
        self.scale *= -1
        self.fmax, self.fmax_optima = self.find_fmin(self.obj_fn, self.bnds, stepsize=0.5, niter=100*len(self.bnds))
        self.fmax *= -1 
        self.scale *= -1
        logging.info(f"fmax=f({self.fmax_optima})={self.fmax} found by BasinHopping. Please double check value.")

        self.ptp = self.fmax - self.fmin

class Region(object):
    def plot(self, _plt):
        # By default plotting is disabled
        pass
    def is_within(self, x):
        raise NotImplemented
        
class BallRegion(Region):
    def __init__(self, coordinate, radius, reward_fn):
        self.coordinate = np.array(coordinate)
        self.radius = radius
        self.reward_fn = eval(reward_fn, {'coordinate':self.coordinate, 'radius':self.radius, 'np':np, 'is_within': self.is_within})
        if len(self.coordinate) == 1:
            self.plot = self.plot_1d
        elif len(self.coordinate) == 2:
            self.plot = self.plot_2d
    def is_within(self, x):
        return np.linalg.norm(x-self.coordinate) <= self.radius
    def plot_1d(self, _plt):
        _plt.axvspan(*[self.coordinate-self.radius, self.coordinate+self.radius], alpha=0.25, color='yellow')
    def plot_2d(self, _plt):
        return _plt.Circle(self.coordinate, self.radius, alpha=0.25, color='red', edgecolor='red', fill=True, clip_on=False)

class CubeRegion(Region):
    def __init__(self, coordinate, length, reward_fn):
        self.coordinate = np.array(coordinate)
        self.num_dim = len(self.coordinate)
        self.length = np.array(length)
        self.min_coord = self.coordinate - self.length/2
        self.max_coord = self.coordinate + self.length/2
        self.bnds = [  (l_bnd, u_bnd) for l_bnd, u_bnd in zip(self.min_coord, self.max_coord) ]
        self.reward_fn = eval(reward_fn, {'coordinate':self.coordinate, 'length':self.length, 'np':np, 'is_within': self.is_within})
        if len(self.coordinate) == 1:
            self.plot = self.plot_1d
        elif len(self.coordinate) == 2:
            self.plot = self.plot_2d
    def is_within(self, x):
        return np.all(x <= self.max_coord) and np.all(x >= self.min_coord)
    def plot_1d(self, _plt):
        _plt.axvspan(*[self.min_coord, self.max_coord], alpha=0.25, color='yellow')
    def plot_2d(self, _plt):
        return _plt.Rectangle(self.coordinate - np.array([self.length/2, self.length/2]), self.length, self.length, alpha=0.25, color='red', edgecolor='red', fill=True, clip_on=False)

class TargetedAttack(BayesOptEnv):
    def __init__(self, region_cls, region_param, **kwargs):

        Region = globals()[region_cls]
        self.region = Region(**region_param)
        super().__init__(reward_fn=self.region.reward_fn, **kwargs)

        # Validate the dimensions
        assert(self.region.num_dim == self.wrapped_fn.num_dim)

        # Declare logging for success
        self.metrics_log_fns += [ self.log_success ]

    def run(self, algorithms, **kwargs):

        self.bo.plot_params['region'] = self.region
        self.bo.additional_plots.append( (self.reward_fn, 'y--', 'Reward Fn') )
        self.bo.additional_plots.extend(algorithms.get_additional_plots())
        
        x = self.x0
        for i in range(self.n_iter+self.initial_design_numdata):
            x, rewards, done, info = self.step(algorithms.perform_attack(x[0]))
            self.render(**self.render_kwargs)
            
        self.bo.finalize()
    def create_fn(self, obj_fn, random_state, **fn_params):
        return NoisyFunction(obj_fn=obj_fn, random_state=random_state, **fn_params)
    def reset(self):
        ret = super().reset()
        # Additional Metrics here
        self.metrics_dict['success_count'] = 0
        return ret

    def log_success(self, next_eval, t, **kwargs):

        if self.region.is_within(next_eval):
            self.metrics_dict['success_count'] += 1

        return {
            'success_rate' : float(self.metrics_dict['success_count']/(t+1)),
            'success_count' : self.metrics_dict['success_count']
        }

class NoisyHpolibFunction(NoisyFunction):
    def __init__(self, hpo_fn_ref, bnds=None, **kwargs):
        self.hpo_fn = getattr(synthetic_functions, hpo_fn_ref)()
        self.info = self.hpo_fn.get_meta_information()
        if bnds == None:
            bnds = self.info['bounds']
        super().__init__(obj_fn=self.hpo_fn, bnds=bnds, **kwargs)
    def init_fmin(self):
        self.fmin = self.info['f_opt']
        self.fmin_optima = self.info['optima']
        logging.info(f"fmin=f({self.fmin_optima})={self.fmin} from HPO.")

        # This is used to find fmax to find ptp
        self.scale *= -1
        self.fmax, self.fmax_optima = self.find_fmin(self.obj_fn, self.bnds, stepsize=0.5, niter=100*len(self.bnds))
        self.fmax *= -1 
        self.scale *= -1
        logging.info(f"fmax=f({self.fmax_optima})={self.fmax} found by BasinHopping. Please double check value.")

        self.ptp = self.fmax - self.fmin

# the noise is turned off
class RobotPushingFunction(Function):
    def __init__(self, num_dim, random_state, **kwargs):
        # Only 2 options, either 
        fn_lookup = {
            3: robot_pushing.robot_push_3d,
            4: robot_pushing.robot_push_4d
        }
        assert(num_dim in fn_lookup)

        # Bnds
        # xmin = [-5; -5; 1; 0];
        # xmax = [5; 5; 30; 2*pi];
        bnds = [
            [-5, 5],
            [-5, 5],
            [1, 30],
            [0, 2*np.pi]
        ]
        bnds = bnds[:num_dim]

        # Dict mapping
        param = ['rx', 'ry', 'steps', 'init_angle']
        param = param[:num_dim]

        #if num_dim == 4:

        # Generate gx gy using the original formula
        # gpos = 10 .* rand(1, 2) - 5;
        gpos = 10. * random_state.rand(1,2) - 5
        gpos = gpos.reshape(-1)
        gpos_dict = dict(zip(['gx', 'gy'], gpos))

        self.target_push_fn = fn_lookup[num_dim]
        self.num_dim = num_dim

        robot_push_fn = lambda x: self.target_push_fn(**{**dict(zip(param, x)), **gpos_dict})

        super().__init__(obj_fn=robot_push_fn, bnds=bnds, random_state=random_state, **kwargs)
    def init_fmin(self):
        
        self.fmin = 0
        self.fmin_optima = None
        logging.info(f"fmin=0, no optima; real dataset has no known fmin_optima.")

        self.fmax_optima = None
        if self.num_dim == 3:
            self.fmax = 14.1421356237
        else:
            self.fmax = 18.3847763109
        logging.info(f"fmax={self.fmax} is estimated using domain knowledge; real dataset has no known fmax_optima.")

        self.ptp = self.fmax - self.fmin