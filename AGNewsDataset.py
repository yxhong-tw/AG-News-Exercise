import pandas as pd

from torch.utils.data import Dataset


class AGNewsDataset(Dataset):
    def __init__(self, config, task_name):
        super().__init__()

        self.data = []

        data_df = pd.read_csv(
            filepath_or_buffer=f'{config["AGNews_path"]}/data/{task_name}.csv'
            , encoding='UTF-8')

        for i in range(data_df.shape[0]):
            one_data = {
                'title': data_df['Title'].iloc[i]
                , 'description': data_df['Description'].iloc[i]
                , 'label': data_df['Class Index'].iloc[i]
            }

        self.data.append(one_data)


    def __getitem__(self, index):
        return self.data[index]


    def __len__(self):
        return len(self.data)
