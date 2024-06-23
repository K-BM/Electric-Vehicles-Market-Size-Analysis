import pandas as pd

class DataFrameProcessor:
    def __init__(self, input_file_path):
        self.dataframe = pd.read_csv(input_file_path)
    
    def collect_ev_data(self):
        return self.dataframe
    
    def drop_missing_values(self):
        self.dataframe = self.dataframe.dropna()
        return self.dataframe
