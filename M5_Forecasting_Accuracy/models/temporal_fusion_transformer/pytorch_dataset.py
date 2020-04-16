import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import get_single_col_by_input_type
from utils.utils import extract_cols_from_data_type
from data_formatters.m5 import M5Formatter
from data_formatters.base import DataTypes, InputTypes

class TFTDataset(Dataset, M5Formatter):
    """Dataset Basic Structure for Temporal Fusion Transformer"""

    def __init__(self, 
                 data_df):
        super(ElectricityFormatter, self).__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        # Attribute loading the data
        self.data = data_df.reset_index(drop=True)
        
        self.id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        self.time_col = get_single_col_by_input_type(InputTypes.TIME, self._column_definition)
        self.target_col = get_single_col_by_input_type(InputTypes.TARGET, self._column_definition)
        self.input_cols = [
                            tup[0]
                            for tup in self._column_definition
                            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
                          ]
        self.col_mappings = {
                              'identifier': [self.id_col],
                              'time': [self.time_col],
                              'outputs': [self.target_col],
                              'inputs': self.input_cols
                          }
        self.lookback = self.get_time_steps()
        self.num_encoder_steps = self.get_num_encoder_steps()
        
        self.data_index = self.get_index_filtering()
        
    def get_index_filtering(self):
        
        unique_indices = self.data[self.id_col].unique()
        dict_index = {i: index_name for i, index_name in enumerate(unique_indices)}
        
        return dict_index

    def __len__(self):
        # In this case, the length of the dataset is not the length of the training data, 
        # rather the ammount of unique sequences in the data
        return self.data_index.shape[0]

    def __getitem__(self, idx):
        
        data_index = self.data[self.data[self.id_col] == self.data_index[idx]]
        
        data_map = {}
        for k in self.col_mappings:
            cols = self.col_mappings[k]
            
            if k not in data_map:
                data_map[k] = [data_index[cols].values]
            else:
                data_map[k].append(data_index[cols].values)
                
        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)
        # Shorten target so we only get decoder steps
        data_map['outputs'] = data_map['outputs'][self.num_encoder_steps:, :]
        
        active_entries = np.ones_like(data_map['outputs'])
        if 'active_entries' not in data_map:
            data_map['active_entries'] = active_entries
        else:
            data_map['active_entries'].append(active_entries)
                
        return data_map['inputs'], data_map['outputs'], data_map['active_entries']