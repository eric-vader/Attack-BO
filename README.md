# Adversarial Attacks on Gaussian Process Bandits

Authors:

1. Eric Han
1. Jonathan Scarlett

This repository is accompanied by the ICML 2022 publication, [arXiv Pre-Print](https://arxiv.org/abs/2110.08449).

## Acknowledgements

1. The code included in `attack_bo/robot_pushing` is taken and adapted from [Max-value Entropy Search/test_functions/python_related](https://github.com/zi-w/Max-value-Entropy-Search/tree/master/test_functions/python_related). The paper accompanying the code is [Max-value entropy search for efficient Bayesian Optimization](https://dl.acm.org/doi/10.5555/3305890.3306056). We have included the code in our repository for ease of setup. This part of the code is copyrighted by the original authors under MIT Licence; See `attack_bo/robot_pushing/LICENSE.txt`.

We acknowledge the packages on which this code is built; the list of packages can be found in `attack_bo.yml`.

## Repository Structure

The code repository is structured as follows:

* `attack_bo/`: Folder containing our code
* `sample/`: Folder containing some sample configurations of `param_file` are given below. Note that we ran the experiments with different random seeds, algorithms, and datasets; this folder is a small sample of the configurations.
* `MLproject`: MLflow project file, we are using mlflow to help organize our experiments
* `README.md`: This readme
* `attack_bo.yml`: Conda file used internally by the MLflow project, which includes all of the packages used.
* `config.yml`: Configuration file that configures the Logger, Mlflow service, Directories, and Daemon.

## Setup

We implemented all algorithms in `Python 3.8.10`; See `attack_bo.yml` for more details.
The Python environments are managed using Conda, and experiments are managed using [MLflow](https://www.mlflow.org), allowing convenient experiments management.

Minimum System requirements:

* `Linux 5.13.12-200`
* `conda 4.10.1`
* `mlflow, version 1.16.0`

Prepare your environment:

1. [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. [Install MLflow](https://mlflow.org/) either on your system or in the base environment of Conda, usually `$ pip install mlflow`

## Running

Check your installation by running the following command in the base directory; this command runs with the test configuration `default.yml`, you should see something similar as output:
```
$ mlflow run .
2021/09/03 15:36:22 INFO mlflow.projects.utils: === Created directory /tmp/tmpx6__r5lp for downloading remote URIs passed to arguments of type 'path' ===
2021/09/03 15:36:22 INFO mlflow.projects.backend.local: === Running command 'source /usr/bin/../etc/profile.d/conda.sh && conda activate mlflow-f1fb335a86e1c806a3a1f640c649cec45922a17b 1>&2 && python attack_bo/main.py /home/Author/Workspace/attack-bo/sample/default.yml' in run with ID '5442c05f976545b98bffed1af4ab5bef' === 
03/09/2021 15:36:23 [INFO    ] [config.py:86] Config[logging] Logging initalized
03/09/2021 15:36:23 [INFO    ] [config.py:96] Config[mlflow_logging] initalized
03/09/2021 15:36:23 [INFO    ] [config.py:98] Config[mlflow_logging] paths uploaded to the server - ['/tmp/tmptroeh86t']
03/09/2021 15:36:23 [INFO    ] [config.py:54] Directory[data] initalized at /tmp/tmptroeh86t/data
03/09/2021 15:36:23 [INFO    ] [config.py:54] Directory[run] initalized at /tmp/tmptroeh86t/run
03/09/2021 15:36:23 [INFO    ] [config.py:54] Directory[cache] initalized at /home/Author/cache
03/09/2021 15:36:23 [INFO    ] [config.py:54] Directory[job] initalized at /home/Author/mlflow-jobs
03/09/2021 15:36:23 [INFO    ] [config.py:54] Directory[daemon] initalized at /dev/shm
03/09/2021 15:36:24 [INFO    ] [datasets.py:401] fmin=f([[-0.85636275]])=-50.460804470465135 found by BasinHopping. Please double check value.
03/09/2021 15:36:24 [INFO    ] [datasets.py:408] fmax=f([[-0.59772549]])=30.719043378478965 found by BasinHopping. Please double check value.
03/09/2021 15:36:24 [INFO    ] [gp.py:47] initializing Y
03/09/2021 15:36:24 [INFO    ] [gp.py:96] initializing inference method
03/09/2021 15:36:24 [INFO    ] [gp.py:105] adding kernel and likelihood as parameters
Optimization restart 1/10, f = 16.370402764686162
Optimization restart 2/10, f = 16.370402745451592
Optimization restart 3/10, f = 16.370402773364546
Optimization restart 4/10, f = 16.37040276092369
Optimization restart 5/10, f = 16.370402750753108
Optimization restart 6/10, f = 16.370402759751517
Optimization restart 7/10, f = 16.37040278509204
Optimization restart 8/10, f = 16.37040280057669
Optimization restart 9/10, f = 16.370402779027035
Optimization restart 10/10, f = 16.370402766212962
03/09/2021 15:36:26 [INFO    ] [datasets.py:315] Matern Kernel updated to var-[122606.67075662], ls-[0.99319659]
Using a model defined by the used.
03/09/2021 15:36:26 [INFO    ] [algorithms.py:16] {'hash': 'c491d073dbabd3063ec04a0a4f53933851bc6f8b', 'type_hash': 'e6ec994f4f06a0faae4b5e902dd75dd030240567', 'random_seed': 0, 'datasets': <datasets.TargetedAttack object at 0x7f4bd36e8fa0>}
03/09/2021 15:36:26 [INFO    ] [main.py:16] Platform: uname_result(system='Linux', node='anon.author.com', release='5.13.12-200.fc34.x86_64', version='#1 SMP Wed Aug 18 13:27:18 UTC 2021', machine='x86_64', processor='x86_64')
03/09/2021 15:36:26 [INFO    ] [main.py:17] Processor: x86_64
03/09/2021 15:36:26 [INFO    ] [main.py:18] Python: 3.8.10/CPython
03/09/2021 15:36:26 [INFO    ] [main.py:21] Blas Library: ['cblas', 'blas', 'cblas', 'blas']
03/09/2021 15:36:26 [INFO    ] [main.py:22] Lapack Library: ['lapack', 'blas', 'lapack', 'blas', 'cblas', 'blas', 'cblas', 'blas']
03/09/2021 15:36:26 [INFO    ] [main.py:26] Temporary Directories: TEMP=/tmp/tmptroeh86t
03/09/2021 15:36:26 [INFO    ] [main.py:27] Host Name: anon.author.com
nothing to optimize
Optimization restart 1/5, f = 16.370402766212962
nothing to optimize
Optimization restart 2/5, f = 16.370402766212962
nothing to optimize
Optimization restart 3/5, f = 16.370402766212962
nothing to optimize
Optimization restart 4/5, f = 16.370402766212962
nothing to optimize
Optimization restart 5/5, f = 16.370402766212962
```

You may run the configurations by using the command to specify the `param_file` configuration, for example:
```
$ mlflow run . -P param_file=sample/1d/clipping_attack.yml
```

### Visualization
In order to visualize the experiment, you can run `mlflow ui` and click on the experiments to visualize the metrics and experiment.

Our experiments include (optional, available for 1D and 2D) live visualization, which can be turned on by using the following inside the `param_file`:
```yaml
    render_kwargs:
      fps: 1
      mode: live
```

Alternatively, you may turn off the live visualization and instead generate the visualization via `mlflow ui`
```yaml
    render_kwargs:
      fps: 1
      mode: file
```
