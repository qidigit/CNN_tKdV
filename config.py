import torch
import numpy as np
import yaml
import os

class ProjectConfig:
    def __init__(self, cfg_path=None, exp_dir=None):
        self.WINDOW_WIDTH = None
        self.WINDOW_HEIGHT = None
        self.WINDOW_SIZE = None
        self.T_START = None
        self.DIL_M = None
        self.STRIDE = None
        self.IN_CHANNELS = None
        self.BATCH_SIZE = None
        self.FOLDER = None
        self.LABELS = None 
        self.N_CLASSES = None
        self.WEIGHTS = None
        self.CACHE = None
        self.DATASET = None
        self.model_final = None
        self.MAIN_FOLDER = None
        self.DATA_FILE = None
        self.TEST_FILE = None
        self.LABEL_FOLDER = None
        self.TRAIN_PATH = None
        self.PRED_PATH = None
        self.CHECK_PATH = None
        self.N_LAYERS = None
        self.LOSS_TYPE = None
        self.KLD_TEMP = None
        self.LOSS_WEIGHT = None
        self.precision = None
        
        # Load a default config initially
        self.init_paths(cfg_path='./experiment/cfg.yml', exp_dir='./experiment/')

    def init_paths(self, cfg_path=None, exp_dir=None):
        self.cfg_path = cfg_path
        self.exp_dir = exp_dir
        self.output_dir = self.exp_dir + 'output/'
        self.load_config()

    def load_config(self):
        # Load config from the config file path
        with open(self.cfg_path, 'r') as cfg_stream:
            data = yaml.load(cfg_stream)

        # parse this and save the items to their corresponding values
        self.WINDOW_WIDTH = data['model']['window_size']
        self.WINDOW_HEIGHT = data['model']['time_size']
        self.T_START = data['model']['time_start']
        self.WINDOW_SIZE = (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        self.DIL_M = data['model']['dil_size']
        self.STRIDE = data['model']['stride']
        self.IN_CHANNELS = data['model']['in_channels']
        self.BATCH_SIZE = data['model']['batch_size']
        self.FIL_SIZE = np.asarray(data['model']['kernel_size'])
        self.N_LAYERS = data['model']['n_layers']
        
        self.LOSS_TYPE = data['model']['loss_type']
        self.KLD_TEMP = np.asarray(data['model']['kld_temp'], dtype=np.float32)
        self.LOSS_WEIGHT = data['model']['loss_weight']

        self.FOLDER = data['data_folder']

        self.LABELS = data['labels']
        self.N_CLASSES = len(self.LABELS)
        self.WEIGHTS = torch.from_numpy(np.asarray(data['class_weights'], dtype=np.float32))

        self.CACHE = data['cache']
        self.DATASET = data['data']['dataset']

        self.CHECK_PATH = data['model_checkpoint']
        self.TRAIN_PATH = data['data']['train_path']
        self.PRED_PATH =  data['data']['pred_path']
        self.MAIN_FOLDER = data['data']['dataset_path']
        self.DATA_FILE = self.MAIN_FOLDER + data['data']['train_file'] + '.' + data['data']['data_format']
        self.TEST_FILE = self.MAIN_FOLDER + data['data']['test_file']  + '.' + data['data']['data_format']
        self.LABEL_FOLDER = self.MAIN_FOLDER + data['data']['label_folder'] + '{}.mat'
        
        self.model_final = data['model_final_path']
        self.precision = data['data']['precision']

# create an object
cfg = ProjectConfig()
