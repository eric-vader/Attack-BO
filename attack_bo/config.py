#!/usr/bin/env python3
from pid import PidFile
import tempfile
import importlib
import os
import yaml
import logging
import time
import mlflow
import shutil

CONFIG_YML = "config.yml"

# https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons
def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

@singleton
class Config:
    def __init__(self):
        self.configs = {}
        self.dynamic_paths = {
            "TEMP" : tempfile.mkdtemp()
        }
        config_dict = yaml.load(open(CONFIG_YML), yaml.FullLoader)
        for sub_module, sub_modules_dict in config_dict.items():
            Clazz = getattr(importlib.import_module('config'), sub_module)
            for config_key, kwargs in sub_modules_dict.items():
                self.configs[config_key] = Clazz(dynamic_paths=self.dynamic_paths, config_key=config_key, **kwargs)
    def finalize(self):
        for k,v in self.configs.items():
            v.finalize()
        for k,v in self.dynamic_paths.items():
            shutil.rmtree(v)

class ConfigItem:
    def __init__(self, config_key, **kwargs):
        self.class_type = type(self).__name__
        self.config_key = config_key
    def finalize(self):
        pass

class Directory(ConfigItem):
    def __init__(self, dynamic_paths, path, exist_ok, **kwargs):
        super().__init__(**kwargs)
        self.path = os.path.expanduser(os.path.join(*path))
        self.path = self.path.format(**dynamic_paths)
        os.makedirs(self.path, exist_ok=exist_ok)
        logging.info(f"Directory[{self.config_key}] initalized at {self.path}")
    def get_path(self, filename):
        return os.path.join(self.path, filename)
    def does_exist(self, filename):
        return os.path.isfile(self.get_path(filename))

# logging_type can either be 'DEFAULT' or 'SERVER'
class Logger(Directory):
    def __init__(self, env_key, log_fmt, date_fmt, file_logging_lvls, **kwargs):
        super().__init__(**kwargs)    
        logging_type = os.getenv(env_key, 'DEFAULT')

        log_formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if root_logger.handlers:
            console = root_logger.handlers[0]  # we assume the first handler is the one we want to configure
        else:
            console = logging.StreamHandler()
            root_logger.addHandler(console)
        console.setFormatter(log_formatter)
        
        for file_logging_lvl in file_logging_lvls:
            file_handler = logging.FileHandler(self.get_path(f"{file_logging_lvl}.log"))
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(getattr(importlib.import_module('logging'), file_logging_lvl))
            root_logger.addHandler(file_handler)

        if logging_type == 'SERVER':
            root_logger.handlers.remove(console)
        else:
            assert(logging_type=='DEFAULT')
        logging.info(f"Config[{self.config_key}] Logging initalized")
    def finalize(self):
        logging.shutdown()
    
class MlflowLogger(ConfigItem):
    def __init__(self, dynamic_paths, retry_count, delay, paths_logged, **kwargs):
        super().__init__(**kwargs)
        self.retry_count = retry_count
        self.delay = delay
        self.allowed_exceptions = (Exception)
        logging.info(f"Config[{self.config_key}] initalized")
        self.paths_logged = [ p.format(**dynamic_paths) for p in paths_logged ]
        logging.info(f"Config[{self.config_key}] paths uploaded to the server - {self.paths_logged}")
    # Taken from https://codereview.stackexchange.com/questions/188539/python-code-to-retry-function
    def retry_fn(self, func, **kwargs):
        for r in range(self.retry_count):
            try:
                result = func(**kwargs)
                return result
            except self.allowed_exceptions as e:
                logging.exception("Exception when contacting server:")
                logging.warning(f"Retrying, waiting for {self.delay} seconds.")
                time.sleep(self.delay)
                pass
    def __call__(self, func_name, **kwargs):
        func = getattr(mlflow, func_name)
        self.retry_fn(func, **kwargs)
    def log_metrics(self, metrics, step):
        self.retry_fn(mlflow.log_metrics, metrics=metrics, step=step)
    def log_params(self, params):
        self.retry_fn(mlflow.log_params, params=params)
    def set_tags(self, tags):
        self.retry_fn(mlflow.set_tags, tags=tags)
    def log_artifacts(self, local_dir, artifact_path=None):
        self.retry_fn(mlflow.log_artifacts, local_dir=local_dir, artifact_path=artifact_path)
    def finalize(self):
        for p in self.paths_logged:
            self.log_artifacts(p)
        
class Daemon(Directory):
    def __init__(self, env_key, **kwargs):
        super().__init__(**kwargs)
        is_daemon_str = os.getenv(env_key, 'FALSE')
        if is_daemon_str != "TRUE" and is_daemon_str != "FALSE":
            raise Exception(f"{env_key} must be either TRUE or FALSE")

        self._is_daemon = is_daemon_str == 'TRUE'
    def is_daemon(self):
        return self._is_daemon
    def lock_pid_file(self, hash):
        logging.info(f"Daemon mode, locking {hash} at {self.path}")
        return PidFile(hash, self.path)