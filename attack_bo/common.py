#!/usr/bin/env python3
from config import Config
import random
import numpy

class Component:
    def __init__(self, hash, type_hash, random_seed, **kwargs):
        self.hash = hash
        self.type_hash = type_hash
        self.config = Config()
        self.mlflow_logger = self.config.configs['mlflow_logging']
        
        self.random_seed = random_seed

        # This is a catch all
        random.seed(random_seed)
        numpy.random.seed(random_seed)

        self.random_state = numpy.random.RandomState(random_seed)

    def prepare(self, **kwargs):
        # cache anything needed
        pass
    def run(self, **kwargs):
        # cache anything needed
        pass