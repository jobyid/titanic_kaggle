import pandas as pd

def replace_Name(test=False):
    if test:
        t_df = pd.read_csv('Test_title.csv')
        t_df.dropna(inplace=True)
        #print(t_df.info())
        return t_df
    else:
        t_df = pd.read_csv('Train_title.csv')
        t_df.dropna(inplace=True)
        #print(t_df.info())
        return t_df

