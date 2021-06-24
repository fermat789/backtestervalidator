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


def create_metric(summary,metric):
    import numpy as np
    metric_ = np.zeros((len(summary),len(summary)))
    for i in range(len(summary)):
        for j in range(len(summary)):
            metric_[i,j] = summary[metric][i]>summary[metric][j]
    return metric_

def extract_uncorrelated_columns(corr,max_corr):
    import numpy as np
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= max_corr:
                if columns[j]:
                    columns[j] = False
    return columns

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
