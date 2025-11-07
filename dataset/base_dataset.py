from __future__ import division

import numpy as np
import pickle

from torch.utils.data import Dataset

import config
import constants

class BaseDataset(Dataset):
    def __init__(self, dataset, hist_length):
        super(BaseDataset, self).__init__()
        
        dataset_path = config.DATASET_FOLDERS[dataset]
        with open(dataset_path, 'rb') as f:
            label = pickle.load(f)

        self.label = label
        self.hist_length = hist_length
        self.len = len(self.label)

    def __getitem__(self, index):

        trunk_path = self.label[index]
        anno = np.load(trunk_path+'.npy')
        Y = {}
        Y['q']  = anno[(config.hist_length-self.hist_length):, :63]
        Y['shape']  = anno[(config.hist_length-self.hist_length):, 63:63+10]
        Y['gender_id']  = anno[(config.hist_length-self.hist_length):, 63+10]
        
        # Add query times for FNO operator mode
        # Generate future query time points (in frame indices)
        if config.model_kind == 'fno_operator':
            if config.query_sampling == 'uniform':
                # Uniform sampling: equally spaced future frames
                Y['query_times'] = np.arange(self.hist_length, config.total_length, dtype=np.float32)
            else:  # 'random'
                # Random non-uniform sampling (for generalization testing)
                query_times = np.sort(np.random.uniform(
                    self.hist_length, config.total_length, 
                    size=config.pred_length
                ))
                Y['query_times'] = query_times.astype(np.float32)
        else:
            # For rollout mode, still provide placeholder
            Y['query_times'] = np.arange(self.hist_length, config.total_length, dtype=np.float32)

        return Y

    def __len__(self):
        return self.len