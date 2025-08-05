import argparse 
import os

def getcfg(file)-> any:
    parser = argparse.ArgumentParser('Training of the Architectures', add_help=True)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default=f'/home/ubuntu/myproject/wcr_neural_kan/configs/{file}')
    args=parser.parse_args()
    return args.config_file