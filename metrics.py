import numpy as np


def get_calmar(pl):
    return calmar(pl)
def get_n_trades(position):
    return np.nan_to_num(np.abs(np.diff(position))).sum()/2
def get_avg_trade(pl,position):
    return pl.sum()/get_n_trades(position)
def get_close_to_close_pl(pl,position):
    mask= np.where(np.diff(position)!=0)[0]+1
    return np.diff(pl.cumsum()[mask-1])
def get_pf(pl,position):
        c_t_c = get_close_to_close_pl(pl,position)
        pf_1 = -1*c_t_c[np.where(c_t_c>0)].sum()/c_t_c[np.where(c_t_c<0)].sum()
        pf_2 = -1*c_t_c[np.where(c_t_c>0)].mean()/c_t_c[np.where(c_t_c<0)].mean()
        pf_3 = -1*pl[np.where(pl>0)].mean()/pl[np.where(pl<0)].mean()
        return pf_1,pf_2,pf_3
def get_elementary_stats(pl,position):
    n_trades = get_n_trades(position)
    if n_trades>0:
        avg_trade = get_avg_trade(pl,position)
    if n_trades==0:
        avg_trade = 0
    calmar = get_calmar(pl)
    return avg_trade,n_trades,calmar

def calmar(strategy):
    if max_draw(strategy)>0:
        calmar = np.round(max_profit(strategy)/max_draw(strategy),2)
    if max_draw(strategy)==0:
        calmar = 0
    return calmar

def sharpe(strategies):
    sharpe = strategies.mean()/strategies.std()
    return sharpe

def draw(strategy):
    draw = np.maximum.accumulate(strategy.cumsum())-strategy.cumsum()
    return draw
def max_draw(strategy):
    max_draw =draw(strategy).max()
    return max_draw
def max_profit(strategy):
    max_profit =np.maximum.accumulate(strategy.cumsum()).max()
    return max_profit
