#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 03:36:52 2022

@author: patyukoe
"""
import torch
import argparse
import yaml



from supervisor import Supervisor
from torchvision import transforms
import pandas as pd



def main(args):
    with open(args.config_filename) as f:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        supervisor_config = yaml.safe_load(f)

        train_filename = supervisor_config['data'].get('train_data')
        test_filename = supervisor_config['data'].get('test_data')
        
        trainset=pd.read_csv(train_filename)
        testset=pd.read_csv(test_filename)
        

        supervisor = Supervisor(trainset, testset, device, **supervisor_config)

        supervisor.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    main(args)

