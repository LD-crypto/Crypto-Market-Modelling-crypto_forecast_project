'''
Build a number of derived features using the raw bar data.
Functions to fractionally differentiate the features to make therm stationary.
'''
import numpy as np
import pandas as pd
import os

def getWeights(d,size):
    '''
    Fractionally differentiation. Compute a diff. series using weights of the previous values.
    Xt = SUM[ wk*Xt-k ]
    The weights can be iteratively estimated with this function.
    '''
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w

def fracDiff(series,d,thres=.01):
    '''
    Standard FracDiff (Expanding Window)
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights(d,series.shape[0])
    #2) Determine initial calcs to be skippeds based on weight-loss threshold
    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]):continue # exclude NAs
            df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

def getWeights_FFD(d,thres=1e-5):
    '''
    Get weights up above threshold for fixed window FracDiff (FFD)
    '''
    # thres>0 drops insignificant weights
    w=[1.]
    k=1
    while abs(w[-1]) > thres:
        w_=-w[-1]/k*(d-k+1)
        w.append(w_)
        k+=1
    w=np.array(w[::-1]).reshape(-1,1)
    return w

def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Fixed-Width Window FracDiff
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be an positive fractional, not necessarilt bounded [0,1].
    '''
    #1) Computed weights for the longest series
    w=getWeights_FFD(d,thres)
    width=len(w)-1
    #2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

