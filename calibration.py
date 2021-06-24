from metrics import *
import numpy as np

def find_strategies(model,train,test,min_calmar,min_avg_trade,bpv,symbol,cost_IS,cost_OS,combination):
    #from tqdm import tqdm
    import pandas as pd
    X = pd.DataFrame([])
    if len(train)>0:
        for comb in combination:#hour in hours_list:
            pos_IS = model(train[symbol],comb,bpv,cost_IS).position
            pl_IS = model(train[symbol],comb,bpv,cost_IS).pl
            avg_trade_IS,n_trades_IS,calmar_IS = get_elementary_stats(pl_IS,pos_IS)
            if (avg_trade_IS >min_avg_trade) &(calmar_IS>min_calmar)  &(n_trades_IS >500):
                pos_OS = model(test[symbol],comb,bpv,cost_OS).position
                pl_OS = model(test[symbol],comb,bpv,cost_OS).pl
                avg_trade_OS,n_trades_OS,calmar_OS = get_elementary_stats(pl_OS,pos_OS)
                C = [(comb,n_trades_IS,n_trades_OS,
                      calmar_IS,avg_trade_IS,*get_pf(pl_IS,pos_IS),pl_IS,
                      calmar_OS,avg_trade_OS,*get_pf(pl_OS,pos_OS),pl_OS) ]
                if len(C) >0:
                    X = pd.concat([X,pd.DataFrame(C)],axis = 0)
                    print(comb,n_trades_IS,n_trades_OS,calmar_IS,calmar_OS,np.round(avg_trade_IS,1),np.round(avg_trade_OS,1),symbol)
    if len(X)>0:
        X.columns = ['comb','n_trades_IS','n_trades_OS','Calmar_IS','Avg_trade_IS','PF_1_IS','PF_2_IS','PF_3_IS','IS_strat','Calmar_OS','Avg_trade_OS','PF_1_OS','PF_2_OS','PF_3_OS','OS_strat']#,'IS_pos','Calmar_OS','Avg_trade_OS','OS_strat','OS_pos']
        X = X.reset_index().drop('index',axis = 1)
        #pd.DataFrame.to_csv(X.iloc[:,:7],symbol+'_new_model_2_2007_2016.csv')
    return X



def f_find_strategies(args):
    a, b, c, d, e,f,g,h,i,j = args[0] , args[1] , args[2] , args[3], args[4],args[5],args[6],args[7],args[8],args[9]
    #print(args[2])
    return find_strategies(a,b,c,d,e,f,g,h,i,j)

