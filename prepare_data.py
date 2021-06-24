def load_data_MC(file):
    import pandas as pd
    df = pd.read_csv('E:\python\data_MC\\\\'+file)
    df.columns = ['DATE','TIME','OPEN','HIGH','LOW','CLOSE','VOLUME']
    df['TIME'] = pd.to_timedelta(df.TIME)
    df['DATE'] = pd.to_datetime(df.DATE,dayfirst=True)
    df['DATETIME'] = df.DATE+df.TIME
    df = df.drop('VOLUME',axis = 1)
    df = df[df.DATETIME.dt.minute == 0]
    df = df.set_index('DATETIME')
    df['HOURS'] = df.index.hour
    return df

def train_test_split(df,years_train,years_test):
    mask_train = df.index.year.isin(years_train)
    mask_test = df.index.year.isin(years_test)
    df_train = df[mask_train]#.reset_index().drop('index',axis = 1)
    df_test = df[mask_test]#.reset_index().drop('index',axis = 1)
    return df_train,df_test
