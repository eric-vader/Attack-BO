from GPyOpt.methods import BayesianOptimization
import GPyOpt
import numpy as np
import time
from GPyOpt.util.general import best_value, normalize
import matplotlib.pyplot as plt
from pylab import savefig
import logging
import maxvar

# This only concerns plotting.
GRID_SIZE = 200

class GpyWrapper(BayesianOptimization):
    def __init__(self, attack_init, acquisition_weight_fn=None, plot_params={}, **kwargs):

        # Check if acq. fn is set, set the weight.
        # https://github.com/SheffieldML/GPyOpt/blob/0be0508f00934043815dd46b9a331e3847070aae/GPyOpt/acquisitions/LCB.py#L7
        if acquisition_weight_fn != None:
            self.acquisition_weight_fn = eval(acquisition_weight_fn)
            # Make sure weight is set properly.
            assert(not 'acquisition_weight' in kwargs)
            kwargs['acquisition_weight'] =  self.acquisition_weight_fn(0)

        # Super hacks TODO
        if kwargs['acquisition_type'] == "MaxVar":
            kwargs['acquisition'] = maxvar.AcquisitionMaxVar()

        super().__init__(**kwargs)

        # Super hacks TODO
        if kwargs['acquisition_type'] == "MaxVar":
            kwargs['acquisition'].delayed_init(self.model, self.space, self.acquisition_optimizer, exploration_weight=kwargs['acquisition_weight'])
        
        self.attack_init = attack_init

        # This determines the number of iterations
        self.num_iter = 0
        self.adv_noises = []
        # Now we also fix the suggested sample
        self.suggested_sample = self.X[self.num_iter].reshape(1,-1)

        # For plotting only
        self.additional_plots = []
        self.plot_params = plot_params
        self.target_bounds = []
        # Precompute some grids
        bounds = self.acquisition.space.get_bounds()
        if len(bounds) <= 2:
            x_grid = [ np.linspace(ax_bnds[0], ax_bnds[1], GRID_SIZE) for ax_bnds in bounds ]
            self.raw_x_grid = x_grid
            x_grid = np.meshgrid(*x_grid)
            self.x_grid = np.hstack([ x_i.reshape(-1, 1) for x_i in x_grid ])
            self.fx_cache = {}

            if self.f.num_dim == 1:
                self.plot_fn  = self.plot_1d
            else:
                self.plot_fn  = self.plot_2d
        else:
            def no_plot_warning(**kwargs):
                logging.error("Cannot show num_dim > 2, please turn off plotting.")

            self.plot_fn = no_plot_warning 

    def suggest_next(self):
        try:
            self._update_model(self.normalization_type)
        except np.linalg.linalg.LinAlgError:
            print("np.linalg.linalg.LinAlgError")
            self.suggested_sample = None

        if ((len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)):
            self.suggested_sample = None

        self.suggested_sample = self._compute_next_evaluations()

    def opt_step(self, adv_noise, eps = 1e-8, context = None, verbosity=False, save_models_parameters= True, report_file = None, evaluations_file = None, models_file=None):
        
        if self.num_iter < self.initial_design_numdata:
            
            # ignore adv noise
            if not self.attack_init:
                adv_noise = 0

            self.Y[self.num_iter] += adv_noise
            self.num_iter += 1
            # Select next suggest sample
            if self.num_iter < self.initial_design_numdata:
                # We 'fake' the next suggested
                self.suggested_sample = self.X[self.num_iter].reshape(1,-1)
            else:
                # The case where we really update the model and suggest the next sample
                # This means it reached the end.
                if self.objective is None:
                    raise InvalidConfigError("Cannot run the optimization loop without the objective function")

                # --- Save the options to print and save the results
                self.verbosity = verbosity
                self.save_models_parameters = save_models_parameters
                self.report_file = report_file
                self.evaluations_file = evaluations_file
                self.models_file = models_file
                self.model_parameters_iterations = None
                self.context = context

                # --- Check if we can save the model parameters in each iteration
                if self.save_models_parameters == True:
                    if not (isinstance(self.model, GPyOpt.models.GPModel) or isinstance(self.model, GPyOpt.models.GPModel_MCMC)):
                        print('Models printout after each iteration is only available for GP and GP_MCMC models')
                        self.save_models_parameters = False

                # --- Setting up stop conditions
                self.eps = eps

                # --- Initial function evaluation and model fitting
                if self.X is not None and self.Y is None:
                    self.Y, cost_values = self.objective.evaluate(self.X)
                    if self.cost.cost_type == 'evaluation_time':
                        self.cost.update_cost_model(self.X, cost_values)

                # --- Initialize iterations and running time
                self.time_zero = time.time()
                self.cum_time  = 0
                self.num_acquisitions = 0
                self.Y_new = self.Y
                self.suggest_next()
        else:    
            # Eval, normal points

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective_noise(adv_noise)

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1
            self.acquisition.exploration_weight = self.acquisition_weight_fn(self.num_acquisitions)

            if True:
                print(f"{self.num_iter} - num acquisition: {self.num_acquisitions}, time elapsed: {self.cum_time:.2f}s")
            self.num_iter += 1            
            self.suggest_next()
        
        return self.suggested_sample

    def finalize(self):
        # --- Stop messages and execution time
        self._compute_results()

        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)
    def evaluate_objective_noise(self, adv_noise):
        self.adv_noises.append(adv_noise)
        self.Y_new, cost_new = self.objective.evaluate(np.array([self.suggested_sample]))
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        self.Y = np.vstack((self.Y,self.Y_new + adv_noise))
    def plot_acquisition_noblock(self, fig, filename=None, opt='min', label_x=None, label_y=None):

        self.fig = fig

        if self.model.model is None:
            from copy import deepcopy
            model_to_plot = deepcopy(self.model)
            if self.normalize_Y:
                Y = normalize(self.Y, self.normalization_type)
            else:
                Y = self.Y
            model_to_plot.updateModel(self.X, Y, self.X, Y)
        else:
            model_to_plot = self.model

        self.plot_fn(model=model_to_plot.model,
                                Xdata = model_to_plot.model.X,
                                Ydata = model_to_plot.model.Y,
                                acquisition_function = self.acquisition.acquisition_function,
                                suggested_sample = [self.suggested_sample],
                                filename=filename,
                                label_x = label_x,
                                label_y = label_y, 
                                f=self.f.obj_fn,
                                additional_plots=self.additional_plots,
                                opt=opt,
                                fmin=self.f.fmin,
                                fmin_optima=self.f.fmin_optima,
                                **self.plot_params)

    def plot_1d(self, model, Xdata, Ydata, acquisition_function, suggested_sample, 
                     filename=None, label_x=None, label_y=None, color_by_step=True, f=None, additional_plots=[], 
                     fmin=None, fmin_optima=None, opt='min', region=None, is_cache=True):

        # Inverse Axis
        ax_i = 1
        if opt == 'max':
            ax_i = -1

        bounds = self.acquisition.space.get_bounds()

        acqu = acquisition_function(self.x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        m, v = model.predict(self.x_grid)
        m = ax_i * m
        
        plt.clf()
        plt.cla()
        #model.plot_density(bounds[0], alpha=.5)

        if region != None:
            region.plot(plt)
        
        plt.plot(self.x_grid, m, 'k-',lw=1,alpha = 0.6)
        
        _lower = m-1.96*np.sqrt(v)
        _upper = m+1.96*np.sqrt(v)

        lower = _lower.reshape(1,-1)[0]
        upper = _upper.reshape(1,-1)[0]
        plt.plot(self.x_grid, _lower, 'k-', alpha = 0.2)
        plt.plot(self.x_grid, _upper, 'k-', alpha = 0.2)
        plt.fill_between(self.x_grid.reshape(1,-1)[0], lower, upper, facecolor='royalblue', interpolate=True,alpha=.25)

        plots = []
        if f != None:
            plots.append([f, 'g-', '$f(x)$'])
        plots.extend(self.additional_plots)

        for plot_fn, plot_line, plot_lbl in plots:

            if plot_lbl in self.fx_cache:
                _plot_fx, _plot_fx_min, _plot_fx_max = self.fx_cache[plot_lbl]
            else:
                _plot_fx = np.apply_along_axis(plot_fn, 1, self.x_grid)
                _plot_fx_min = min(_plot_fx)
                _plot_fx_max = max(_plot_fx)
                if is_cache:
                    self.fx_cache[plot_lbl] = _plot_fx, _plot_fx_min, _plot_fx_max

            plot_fx = ax_i * _plot_fx.reshape(1,-1)[0]
            lower = np.append(lower, _plot_fx_min)
            upper = np.append(upper, _plot_fx_max)
            plt.plot(self.x_grid, plot_fx, plot_line,lw=2, label=plot_lbl)

        plt.plot(Xdata, ax_i * Ydata, 'r.', markersize=10)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        
        factor = max(upper)-min(lower)
        plt.plot(self.x_grid,0.2*factor*acqu_normalized-abs(min(lower))-0.25*factor, 'r-',lw=2,label ='Acquisition Fn')

        ylim = (min(lower)-0.25*factor,  max(upper)+0.2*factor)

        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.ylim(*ylim)
        
        plt.legend(loc='upper right')

        savefig(filename)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def plot_2d(self, model, Xdata, Ydata, acquisition_function, suggested_sample, 
                     filename=None, label_x=None, label_y=None, color_by_step=True, f=None, additional_plots=[], 
                     fmin=None, fmin_optima=None, opt='min', region=None, is_cache=True):

        bounds = self.acquisition.space.get_bounds()

        plt.clf()
        plt.cla()

        if not label_x:
            label_x = 'X1'

        if not label_y:
            label_y = 'X2'

        n = Xdata.shape[0]
        colors = np.linspace(0, 1, n)
        cmap = plt.cm.Reds
        norm = plt.Normalize(vmin=0, vmax=1)
        points_var_color = lambda X: plt.scatter(
            X[:,0], X[:,1], c=colors, label=u'Observations', cmap=cmap, norm=norm)
        points_one_color = lambda X: plt.plot(
            X[:,0], X[:,1], 'r.', markersize=10, label=u'Observations')
        X1, X2 = self.raw_x_grid
        
        # Cache
        true_f = np.apply_along_axis(f, 1, self.x_grid).reshape((200,200))
        acqu = acquisition_function(self.x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        m, v = model.predict(self.x_grid)

        plots = []
        if f != None:
            plots.append([f, 'g-', f'$f(x)$, fmin={fmin}'])
        plots.extend(self.additional_plots)

        num_plots = len(plots) + 3

        # 1
        plt.subplot(1, num_plots, 1)
        plt.contourf(X1, X2, m.reshape(200,200),100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xdata)
        else:
            points_one_color(Xdata)
        plt.ylabel(label_y)
        plt.title('Posterior mean')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        
        ## 2
        plt.subplot(1, num_plots, 2)
        plt.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xdata)
        else:
            points_one_color(Xdata)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Posterior sd.')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        
        ## 3
        plt.subplot(1, num_plots, 3)
        plt.contourf(X1, X2, acqu_normalized,100)
        plt.colorbar()
        suggested_sample = suggested_sample[0]
        plt.plot(suggested_sample[:,0],suggested_sample[:,1],'m.', markersize=10)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        
        plot_i = 4
        for plot_fn, plot_line, plot_lbl in plots:

            if plot_lbl in self.fx_cache:
                _plot_fx = self.fx_cache[plot_lbl]
            else:
                _plot_fx = np.apply_along_axis(plot_fn, 1, self.x_grid).reshape((200,200))
                self.fx_cache[plot_lbl] = _plot_fx

            ax = plt.subplot(1, num_plots, plot_i)
            plt.xlabel(label_x)
            plt.ylabel(label_y)

            plt.title(plot_lbl)
            cs = plt.contour(X1, X2, _plot_fx, 25)
            #plt.colorbar()
            ax.clabel(cs, inline=True, fontsize=8)

            # temp
            f_opt = np.array(fmin_optima)
            plt.plot(f_opt[:,0],f_opt[:,1],'r.', markersize=10)

            if region != None:
                ax.add_patch(region.plot(plt))

            plot_i += 1

        savefig(filename)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()