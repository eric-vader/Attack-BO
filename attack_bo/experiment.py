#!/usr/bin/env python3
import yaml
import hashlib
import json
import importlib
import os
import copy

from config import Config

hash_args = lambda args: hashlib.sha1(json.dumps(args, sort_keys=True).encode()).hexdigest()

class Experiment:
    def __init__(self, exp_config):

        config = Config()

        # We guess if config is a file or not
        extension = os.path.splitext(exp_config)[-1]

        if not os.path.isfile(exp_config):
            raise FileNotFoundError(f"Cannot experiment file at {exp_config}")

        self.exp_args = yaml.load(open(exp_config), yaml.FullLoader)
        self.hash = hash_args(self.exp_args)
        # Simple checks

        self.sub_module_hashes = {}
        self.sub_module_type_hashes = {}
        self.sub_module_name_lookup = {}
        self.sub_module_instances = {}
        for module,v in self.exp_args.items():
            if type(v) != dict:
                raise Exception(f"Unexpected type for {module}, must be dict")
            if len(v.keys()) != 1:
                raise Exception(f"Must only init one object for every package")

            self.sub_module_hashes[module] = hash_args(v)

            # This is just to calculate the type hash
            v_type = copy.deepcopy(v)
            (_, v_type_kwargs), = v_type.items()
            if 'random_seed' in v_type_kwargs:
                del v_type_kwargs['random_seed']
            self.sub_module_type_hashes[module] = hash_args(v_type)

            (sub_module, kwargs), = v.items()
            self.sub_module_name_lookup[module] = sub_module

            Clazz = getattr(importlib.import_module(module), sub_module)
            self.sub_module_instances[module] = Clazz(hash=self.sub_module_hashes[module], type_hash=self.sub_module_type_hashes[module], **{**kwargs, **self.sub_module_instances})