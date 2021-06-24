
def numpy_fill(arr):
    import numpy as np
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[0]),0)
    np.maximum.accumulate(idx,axis=0, out=idx)
    out = arr[idx]
    return out

def calmar(strategy):
    import numpy as np
    if max_draw(strategy)>0:
        #calmar = np.round(max_profit(strategy)/max_draw(strategy),2)
        calmar = np.round(max_profit(strategy)/max_draw(strategy),2)
    if max_draw(strategy)==0:
        calmar = 0
    return calmar

def sharpe(strategies):
    sharpe = strategies.mean()/strategies.std()
    return sharpe

def draw(strategy):
    import numpy as np
    draw = np.maximum.accumulate(strategy.cumsum())-strategy.cumsum()
    return draw
def max_draw(strategy):
    max_draw =draw(strategy).max()
    return max_draw
def max_profit(strategy):
    import numpy as np
    max_profit =np.maximum.accumulate(strategy.cumsum()).max()
    return max_profit

class model_bias_TF:
    import numpy as np
    def __init__(self,df,comb,bpv,cost):
        import numpy as np
        import pandas as pd
        self.df = df
        self.bpv = bpv
        self.cost = cost
        self.timeok_interval = comb[0]
        self.TIMEOK =df.HOURS.isin(self.timeok_interval)
        self.typology = comb[2]
        self.par = comb[1]
        self.position = self.get_position()
        self.pl = self.get_pl()#-self.get_cost()
        self.n_trades = self.get_n_trades()
        self.avg_trade = self.get_avg_trade()
        self.pf = self.get_pf()
        self.calmar = self.get_calmar()

    def get_cond_entry(self):
        import numpy as np
        import pandas as pd
        if self.typology == 'MR':
            cond_long = (self.TIMEOK.shift(1).values) & (self.df['CLOSE'].shift(1)<self.df['CLOSE'].shift(self.par+1)) & (self.df['CLOSE'].shift(1)<self.df['CLOSE'].shift(2+1))
            cond_short = (self.TIMEOK.shift(1).values == False) & (self.df['CLOSE'].shift(1)>self.df['CLOSE'].shift(self.par+1)) & (self.df['CLOSE'].shift(1)>self.df['CLOSE'].shift(2+1))
        if self.typology == 'BO':
            cond_long = (self.TIMEOK.shift(1).values) & (self.df['CLOSE'].shift(1)>self.df['CLOSE'].shift(self.par+1))  & (self.df['CLOSE'].shift(1)>self.df['CLOSE'].shift(2+1))
            cond_short =(self.TIMEOK.shift(1).values == False) & (self.df['CLOSE'].shift(1)<self.df['CLOSE'].shift(self.par+1))  & (self.df['CLOSE'].shift(1)<self.df['CLOSE'].shift(2+1))
        if self.typology == 'TF_MR':
            cond_long = (self.TIMEOK.shift(1).values) & (self.df['CLOSE'].shift(1)<self.df['CLOSE'].shift(self.par+1)) & (self.df['CLOSE'].shift(1)>self.df['CLOSE'].shift(2+1))
            cond_short = (self.TIMEOK.shift(1).values == False) & (self.df['CLOSE'].shift(1)>self.df['CLOSE'].shift(self.par+1)) & (self.df['CLOSE'].shift(1)<self.df['CLOSE'].shift(2+1))
        if self.typology == 'TF_BO':
            cond_long = (self.TIMEOK.shift(1).values) & (self.df['CLOSE'].shift(1)>self.df['CLOSE'].shift(self.par+1)) & (self.df['CLOSE'].shift(1)<self.df['CLOSE'].shift(2+1))
            cond_short = (self.TIMEOK.shift(1).values == False) & (self.df['CLOSE'].shift(1)<self.df['CLOSE'].shift(self.par+1)) & (self.df['CLOSE'].shift(1)>self.df['CLOSE'].shift(2+1))  
        return cond_long , cond_short
    
    def get_position(self):
        import numpy as np
        import pandas as pd
        pos = np.nan_to_num(numpy_fill(np.select([self.get_cond_entry()[0],self.get_cond_entry()[1]],[1,-1],default=np.nan)))
        return pos

    def get_pl(self):
        import numpy as np
        import pandas as pd
        pl =self.bpv*np.nan_to_num(np.select([np.diff(self.position,prepend=np.nan)==0,np.diff(self.position,prepend=np.nan)!=0],[(self.position*self.df.CLOSE.diff()),(self.position*(self.df.CLOSE-self.df.OPEN))]))
        return pl
    
   # def get_cost(self):
        #import numpy as np
        #import pandas as pd
        #return np.where(np.diff(self.position) == 0,0,self.cost/2)
    def get_calmar(self):
        return calmar(self.pl)
    def get_n_trades(self):
        import numpy as np
        return np.nan_to_num(np.abs(np.diff(self.position))).sum()/2
    def get_avg_trade(self):
        return self.get_pl().sum()/self.get_n_trades()

    def get_close_to_close_pl(self):
        import numpy as np
        mask= np.where(np.diff(self.position)!=0)[0]+1
        return np.diff(self.pl.cumsum()[mask-1])
    def get_pf(self):
        import numpy as np
        import pandas as pd
        c_t_c = self.get_close_to_close_pl()
        pf_1 = -1*c_t_c[np.where(c_t_c>0)].sum()/c_t_c[np.where(c_t_c<0)].sum()
        pf_2 = -1*c_t_c[np.where(c_t_c>0)].mean()/c_t_c[np.where(c_t_c<0)].mean()
        pf_3 = -1*self.pl[np.where(self.pl>0)].mean()/self.pl[np.where(self.pl<0)].mean()
        return pf_1,pf_2,pf_3
    def get_elementary_stats(self):
        import numpy as np
        import pandas as pd
        pl = self.pl
        n_trades = self.n_trades
        if n_trades>0:
           avg_trade = self.avg_trade
        if n_trades==0:
            avg_trade = 0
        calmar = self.calmar
        return avg_trade,n_trades,calmar


def find_strategies_model_bias_TF(train,test,min_calmar,min_avg_trade,bpv,symbol,cost_IS,cost_OS,combination):
    import numpy as np
    import pandas as pd
    #from tqdm import tqdm
    X = pd.DataFrame([])
    if len(train)>0:
        for comb in combination:#hour in hours_list:
            TRS = model_bias_TF(train[symbol],comb,bpv,cost_IS)
            TES = model_bias_TF(test[symbol],comb,bpv,cost_OS)
            avg_trade_IS,n_trades_IS,calmar_IS = TRS.get_elementary_stats()
            if (avg_trade_IS >min_avg_trade) &(calmar_IS>min_calmar)  &(n_trades_IS >500): 
                avg_trade_OS,n_trades_OS,calmar_OS = TES.get_elementary_stats()
                pl_IS = TRS.pl
                pl_OS = TES.pl
                C = [(comb,n_trades_IS,n_trades_OS,
                      calmar_IS,avg_trade_IS,*TRS.get_pf(),pl_IS,
                      calmar_OS,avg_trade_OS,*TES.get_pf(),pl_OS) ]
                if len(C) >0:
                    X = pd.concat([X,pd.DataFrame(C)],axis = 0)
                    print(comb,n_trades_IS,n_trades_OS,calmar_IS,calmar_OS,np.round(avg_trade_IS,1),np.round(avg_trade_OS,1),symbol)
    if len(X)>0:
        X.columns = ['comb','n_trades_IS','n_trades_OS','Calmar_IS','Avg_trade_IS','PF_1_IS','PF_2_IS','PF_3_IS','IS_strat','Calmar_OS','Avg_trade_OS','PF_1_OS','PF_2_OS','PF_3_OS','OS_strat']#,'IS_pos','Calmar_OS','Avg_trade_OS','OS_strat','OS_pos']
        X = X.reset_index().drop('index',axis = 1)
        #pd.DataFrame.to_csv(X.iloc[:,:7],symbol+'_new_model_2_2007_2016.csv')
    return X


def f_find_strategies_model_bias_TF(args):
    a, b, c, d, e,f,g,h,i = args[0] , args[1] , args[2] , args[3], args[4],args[5],args[6],args[7],args[8]
    #print(args[2])
    return find_strategies_model_bias_TF(a,b,c,d,e,f,g,h,i)


bpv = {}
bpv['AEX'] = 200
bpv['MT'] =10
bpv['EX'] = 10
bpv['EB'] = 1000
bpv['LF'] = 10
bpv['SW'] = 10
bpv['AD']=100000
bpv['BP']=62500
bpv['BTC']=1
bpv['CC']= 10
bpv['CD']=100000
bpv['CL']= 1000
bpv['CT']= 500
bpv['DX']= 1000
bpv['EU']= 125000
bpv['ES']= 50
bpv['BTP']= 1000
bpv['XG']= 25
bpv['BD']= 1000
bpv['GC']= 100
bpv['HG']= 25000
bpv['HO']= 42000
bpv['JY']=125000
bpv['KC']=375
bpv['NG']= 10000
bpv['NKD']= 5
bpv['NQ']= 20
bpv['OJ']= 150
bpv['PA']= 100
bpv['PL']= 50
bpv['RB']= 42000
bpv['RTY']= 50 
bpv['SB']=1120
bpv['SI']= 5000
bpv['TY'] = 1000

min_avg_trade = {}
min_avg_trade['AEX'] = 150
min_avg_trade['MT'] =100
min_avg_trade['EX'] = 120
min_avg_trade['EB'] = 100
min_avg_trade['LF'] = 100
min_avg_trade['SW'] = 100
min_avg_trade['AD']=90
min_avg_trade['BP'] = 90
min_avg_trade['BTC']=100
min_avg_trade['CC']= 100
min_avg_trade['CD']=90
min_avg_trade['CL']= 100
min_avg_trade['CT']= 120
min_avg_trade['DX']= 80
min_avg_trade['EU']= 80
min_avg_trade['ES']= 50
min_avg_trade['BTP']= 100
min_avg_trade['XG']= 200
min_avg_trade['BD']= 100
min_avg_trade['GC']= 100
min_avg_trade['HG']= 90
min_avg_trade['HO']= 120
min_avg_trade['JY']=90
min_avg_trade['KC']=200
min_avg_trade['NG']= 100
min_avg_trade['NKD']= 100
min_avg_trade['NQ']= 100
min_avg_trade['OJ']= 150
min_avg_trade['PA']= 200
min_avg_trade['PL']= 120
min_avg_trade['RB']= 150
min_avg_trade['RTY']= 100
min_avg_trade['SB']=100
min_avg_trade['SI']= 200
min_avg_trade['TY'] = 100

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

def merge_equity_lines(SUMMARY,symbols,df_train,df_test,frequency):
    import pandas as pd
    import numpy as np
    IS_strat = {}
    OS_strat = {}
    IS_strat = {symbol:pd.DataFrame(np.transpose(SUMMARY[symbol]['IS_strat'].tolist())).set_index(df_train[symbol].index)for symbol in symbols}
    OS_strat = {symbol:pd.DataFrame(np.transpose(SUMMARY[symbol]['OS_strat'].tolist())).set_index(df_test[symbol].index)for symbol in symbols}
    mintime_IS = min([IS_strat[symbol].index.min() for symbol in symbols])
    maxtime_IS = max([IS_strat[symbol].index.max() for symbol in symbols])
    mintime_OS = min([OS_strat[symbol].index.min() for symbol in symbols])
    maxtime_OS = max([OS_strat[symbol].index.max() for symbol in symbols])
    from datetime import datetime
    date_rng_IS = pd.date_range(start=mintime_IS, end=maxtime_IS, freq=frequency)
    date_IS = pd.DataFrame([])
    date_IS['DATETIME']= date_rng_IS
    date_rng_OS = pd.date_range(start=mintime_OS, end=maxtime_OS, freq=frequency)
    date_OS = pd.DataFrame([])
    date_OS['DATETIME']= date_rng_OS
    ALL_TIME_IS = {}
    ALL_TIME_OS = {}
    for symbol in symbols:
        ALL_TIME_IS[symbol]= (pd.merge(date_IS,IS_strat[symbol],how='outer',on = 'DATETIME').set_index('DATETIME').fillna(0))
        ALL_TIME_OS[symbol]= (pd.merge(date_OS,OS_strat[symbol],how='outer',on = 'DATETIME').set_index('DATETIME').fillna(0))
    ALL_CHRONO_IS_strategies  = pd.concat(ALL_TIME_IS,axis = 1)
    ALL_CHRONO_OS_strategies  = pd.concat(ALL_TIME_OS,axis = 1)
    return ALL_CHRONO_IS_strategies,ALL_CHRONO_OS_strategies


def create_uncorrelated_list_of_strategies(SUMMARY,ALL_CHRONO_IS_strategies,corr_IS,symbols,max_corr):
    import pandas as pd
    list_to_keep = {}
    for symbol in symbols:#symbol = symbols[0]
        metric_type = ['Calmar_IS','PF_1_IS','PF_2_IS','PF_3_IS','Avg_trade_IS']
        list_to_keep_tmp = []
        for metric_chosen in metric_type:
            metric = create_metric(SUMMARY[symbol],metric_chosen)
            columns_new = extract_uncorrelated_columns_2(corr_IS[symbol],max_corr,metric)
            to_keep_new = ALL_CHRONO_IS_strategies[symbol].columns.values[columns_new]
            list_to_keep_tmp.append([list(to_keep_new),metric_chosen])
        columns_old = {symbol:extract_uncorrelated_columns(corr_IS[symbol],max_corr) for symbol in symbols}
        to_keep_old = ALL_CHRONO_IS_strategies[symbol].columns.values[columns_old[symbol]]
        list_to_keep_tmp.append([list(to_keep_old),'standard'])
        list_to_keep[symbol] = list_to_keep_tmp
    return list_to_keep


def extract_uncorrelated_columns(corr,max_corr):
    import numpy as np
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= max_corr:
                if columns[j]:
                    columns[j] = False
    return columns


def create_metric(summary,metric):
    import numpy as np
    metric_ = np.zeros((len(summary),len(summary)))
    for i in range(len(summary)):
        for j in range(len(summary)):
            metric_[i,j] = summary[metric][i]>summary[metric][j]
    return metric_

def extract_uncorrelated_columns_2(corr,max_corr,metric): 
    import numpy as np 
    columns = np.full((corr.shape[0],), True, dtype=bool) 
    for i in range(corr.shape[0]): 
        for j in range(i+1, corr.shape[0]): 
            if (corr.iloc[i,j] >= max_corr): 
                if (metric[i,j] == 1): 
                    if columns[j]: 
                        columns[j] = False 
                if (metric[i,j] == 0): 
                    if columns[i]:
                        columns[i] = False 
    return columns

def calc_weighted_strat(ALL_CHRONO_IS_strategies,ALL_CHRONO_OS_strategies,list_to_keep,metric_type,symbols,risk_parity_measure):
    import pandas as pd
    IS_metric = {}
    OS_metric = {}
    for j in range(len(metric_type)):
        ALL_metric_IS = {}
        ALL_metric_OS = {}
        for symbol in symbols:
            ALL_metric_IS[symbol] = ALL_CHRONO_IS_strategies[symbol][list_to_keep[symbol][j][0]]
            ALL_metric_OS[symbol] = ALL_CHRONO_OS_strategies[symbol][list_to_keep[symbol][j][0]]
        IS_metric[metric_type[j]] = pd.concat(ALL_metric_IS,axis = 1)
        OS_metric[metric_type[j]] = pd.concat(ALL_metric_OS,axis = 1)
    weights = {metric:calc_weights(IS_metric[metric],risk_parity_measure) for metric in metric_type}
    OS_metric_w = {metric:weights[metric][0]*OS_metric[metric] for metric in metric_type}
    IS_metric_w = {metric:weights[metric][0]*IS_metric[metric] for metric in metric_type}
    return IS_metric_w,OS_metric_w,weights

def calc_weights(portfolio_of_strategies,metric):
    import numpy as np
    if metric == 'profit':
    #normalize uncorrelated strategies with profit
        weights = np.round(portfolio_of_strategies.cumsum().max(axis = 0).max()/portfolio_of_strategies.cumsum().max(axis = 0))
        portfolio_weighted= portfolio_of_strategies*weights
        #return weights,portfolio_weighted
    if metric == 'st_dev':
    #normalize uncorrelated strategies with profit
        weights = np.round(portfolio_of_strategies.cumsum().std(axis = 0)/portfolio_of_strategies.cumsum().std(axis = 0).max())
        portfolio_weighted= portfolio_of_strategies*weights
        #return weights,portfolio_weighted
    if metric == 'calmar':
    #normalize uncorrelated strategies with profit
        weights = np.round(calmar(portfolio_of_strategies)/calmar(portfolio_of_strategies).max())
        #np.round(calmar(portfolio_of_strategies)/calmar(portfolio_of_strategies).max())
        portfolio_weighted= portfolio_of_strategies*weights
    if metric == 'sharpe':
    #normalize uncorrelated strategies with profit
        weights = np.round(sharpe(portfolio_of_strategies)/sharpe(portfolio_of_strategies).max())
        portfolio_weighted= portfolio_of_strategies*weights
    #return weights,portfolio_weighted
    if metric == 'landolfi':
    #normalize uncorrelated strategies with profit
        weights_1 = calmar(portfolio_of_strategies)/calmar(portfolio_of_strategies).max()
        weights_2 = portfolio_of_strategies.cumsum().max(axis = 0).max()/portfolio_of_strategies.cumsum().max(axis = 0)
        #weights_3 = 1/(count(portfolio_of_strategies)/count(portfolio_of_strategies).max())
        weights = np.round(weights_1*weights_2)#*weights_3)
        #np.round(calmar(portfolio_of_strategies)/calmar(portfolio_of_strategies).max())
        portfolio_weighted= portfolio_of_strategies*weights
    if metric == 'SQN':
    #normalize uncorrelated strategies with profit
        weights = np.round(SQN(portfolio_of_strategies)/SQN(portfolio_of_strategies).max())
        portfolio_weighted= portfolio_of_strategies*weights
    if metric == 'max_draw':
    #normalize uncorrelated strategies with profit
        weights = np.round(max_draw(portfolio_of_strategies).max()/max_draw(portfolio_of_strategies))
        #np.round(calmar(portfolio_of_strategies)/calmar(portfolio_of_strategies).max())
        portfolio_weighted= portfolio_of_strategies*weights
    return weights,portfolio_weighted



