#!/usr/bin/env python3
import platform
import socket
import os
import sys
import logging

import numpy as np

import experiment
from config import Config

def main(mlflow_logger, exp):

    # Machine Related logging
    logging.info('Platform: %s', platform.uname())
    logging.info('Processor: %s', platform.processor())
    logging.info('Python: %s/%s', platform.python_version(), platform.python_implementation())

    # Log LAPACK Information
    logging.info("Blas Library: %s", np.__config__.get_info('blas_opt_info')['libraries'])
    logging.info("Lapack Library: %s", np.__config__.get_info('lapack_opt_info')['libraries'])
    
    # Temp directories logging
    for k,v in Config().dynamic_paths.items():
        logging.info(f"Temporary Directories: {k}={v}")
    logging.info(f"Host Name: {platform.node()}")

    # Should not be modified, anything else should be added to params
    tags = {
        "mlflow.runName"    : f"{exp.sub_module_name_lookup['algorithms']}",
        "host_name"         : f"{socket.gethostname().split('.')[0]}",
        "hash_exe"              : exp.hash,
    }
    mlflow_logger.set_tags(tags)

    # Parameters
    params = exp.sub_module_hashes
    mlflow_logger.log_params(params)
    
    # Prepare all sub modules
    for sub_module, instance in exp.sub_module_instances.items():
        instance.prepare(**exp.sub_module_instances)

    # Collect all submodule objects

    # Run all sub modules
    for sub_module, instance in exp.sub_module_instances.items():
        instance.run(**exp.sub_module_instances)

if __name__ == '__main__':

    exit_status = os.EX_OK

    # Init the Singleton Config
    c = Config()
    mlflow_logger = c.configs['mlflow_logging']
    
    # Ensure that the code cannot fail and log everything...
    try:
        exp = experiment.Experiment(sys.argv[1])

        # Check if running in daemon mode 
        if c.configs['daemon'].is_daemon():
            with c.configs['daemon'].lock_pid_file(exp.hash) as p:
                main(mlflow_logger, exp)
        else:
            main(mlflow_logger, exp)
    except Exception as e:
        mlflow_logger.set_tags({"Error": str(type(e).__name__)})
        logging.exception("Exception")
        exit_status = e
    finally:
        # Clean up, log artifacts and remove directory
        c.finalize()
        sys.exit(exit_status)
        
