'''
These scripts are to construct raw datasets from the crypto market data and sample them into a usable bar structure.
There are functions to then sample events from the bar data, label the data and put on training weights to each sample.
Required packages imported:
import numpy as np
import pandas as pd
import os
'''
import numpy as np
import pandas as pd
import os

def constructRawData(path):
    '''
    Combine all the raw data from the pickle files into a single pandas dataframe
    path is the string name of of the folder containing the data, formatted as r''
    '''
    # iterate through the files and join them into one larger dataframe
    df = pd.read_pickle(path+'0000000000000000000.pickle')
    for filename in os.listdir(path):
        A = pd.read_pickle(path + filename)
        df = A.append(df)
    df = df[:-1000]  # remove the duplicate data
    df['date'] = pd.to_datetime(df.time, unit='s')
    df = df.set_index('date')  # change index to UTC datetime
    df['value'] = df['price'] * df['volume']  # get value
    df = df.iloc[::-1]  # order dataframe from oldest to newest
    return df

def sampleBars(data, method, interval, grain='H'):
    '''
    A holding function that can sample raw data using different sampling methods to create data bars
    :param data: The raw tic data. Must have 'date' as index and 'price', 'volume' and 'value' columns
    :param method: A string which indicates the type of sampling that will be used.
    'time bars' or 'tb' = Constant time sampled bars
    'tic bars' or 'tcb'= Bars sampled every number of tics
    'constant volume bars' or 'cvolb' = Bars sampled after a given volume
    'rolling volume bars' or 'rvolb'
    'constant euro bars' or 'ceurob'
    'rolling euro bars' or 'reurob'
    :param interval: The parameter determining the granularity of the sampling.
    Time bars - is a a string 'D', 'H', etc.
    Tic/Volume/Euro - is a number
    Rolling bars - interval is the window size if 0 then an expanding window used
    :param grain: The granularity of the approximate rolling value. Is a string like the time bars.
    :return: A dataframe of bar sampled market data:
    {index = 'date', columns = 'open', 'high', 'low', 'close', 'wavg', 'volume', 'value'}
    '''
    if method == 'time bars' or method == 'tb':
        df = sampleTimeBars(data=data, interval=interval)
    elif method == 'tic bars' or method == 'tcb':
        df = sampleTicBars(data=data, interval=interval)
    elif method == 'constant volume bars' or method == 'cvolb':
        df = sampleConstantVolumeBars(data=data, interval=interval)
    elif method == 'constant euro bars' or method == 'ceurob':
        df = sampleConstantEuroBars(data=data, interval=interval)
    elif method == 'rolling volume bars' or method == 'rvolb':
        df = sampleRollingVolumeBars(data=data, interval=interval, grain=grain)
    elif method == 'rolling euro bars' or method == 'reurob':
        df = sampleRollingEuroBars(data=data, interval=interval, grain=grain)
    return df

def sampleTimeBars(data, interval):
    '''
    Takes as input a raw dataframe and samples it into consistent time-interval bars.
    interval is a string indicating: second, min, hour, day, etc.
    '''
    df_time = data.resample(interval).apply({'volume': 'sum', 'value': 'sum'})
    df_time['wavg'] = df_time.value/df_time.volume # weighted average of the price over the interval
    df_time['open'] = data.resample(interval).apply({'price': 'first'})
    df_time['high'] = data.resample(interval).apply({'price': 'max'})
    df_time['low'] = data.resample(interval).apply({'price': 'min'})
    df_time['close'] = data.resample(interval).apply({'price': 'last'})
    return df_time

def sampleTicBars(data, interval):
    '''
    :param data: raw dataframe as input
    :param interval: the amount of tics used to construct one bar
    :return: a constant tic sampled bar dataframe
    '''
    # Creat tick bars from sampling every interval number of trades
    num = int(len(data) / interval)
    indx = range(0, num)
    col = ['date', 'open', 'high', 'low', 'close', 'volume', 'value', 'wavg']
    df = pd.DataFrame(index=indx, columns=col)
    for i in range(0, num - 1):
        curind = (i * interval) + interval -1  # current index in loop
        df['date'][i] = data.index[curind]  # the timestamp of the tick data is the date of the last tick
        df['open'][i] = data.price[i*interval]
        df['high'][i] = data.price[i*interval:curind].max()
        df['low'][i] = data.price[i*interval:curind].min()
        df['close'][i] = data.price[curind]
        df['volume'][i] = data.volume[i * interval:curind].sum()
        df['value'][i] = data.value[i*interval:curind].sum()
    df['wavg'] = df.value/df.volume
    df = df.set_index('date')
    return df

def sampleConstantVolumeBars(data, interval):
    '''
    This function will sample bar data from raw df ever constant amount of volume
    :param data: the raw df
    :param interval: the amount of volume traded in a single bar
    :return: a bar sampled df
    '''
    num = int(data.volume.sum()/interval)
    indx = range(0, num)
    col = ['date', 'open', 'high', 'low', 'close', 'volume', 'value', 'wavg']
    df = pd.DataFrame(index=indx, columns=col)
    curind = 0
    volsum = 0  # volume counter
    r = 0 # previous bar index counter
    for i in range(0, len(data)):
        volsum += data.volume[i]
        if volsum > interval:
            df['date'][curind] = data.index[i]  # the timestamp of the tick data is the date of the last tick
            df['open'][curind] = data.price[r]
            df['high'][curind] = data['price'][r:i+1].max()
            df['low'][curind] = data['price'][r:i+1].min()
            df['close'][curind] = data.price[i]
            df['volume'][curind] = volsum
            df['value'][curind] = data['value'][r:i+1].sum()
            curind += 1
            volsum = 0
            r = i
    df['wavg'] = df.value / df.volume
    df = df.set_index('date')
    return df

def sampleConstantEuroBars(data, interval):
    '''
    This function will sample bar data from raw df ever constant amount of value
    '''
    num = int(data.value.sum()/interval)
    indx = range(0, num)
    col = ['date', 'open', 'high', 'low', 'close', 'volume', 'value', 'wavg']
    df = pd.DataFrame(index=indx, columns=col)
    curind = 0
    valsum = 0  # value counter
    r = 0 # previous bar index counter
    for i in range(0, len(data)):
        valsum += data.value[i]
        if valsum > interval:
            df['date'][curind] = data.index[i]  # the timestamp of the tick data is the date of the last tick
            df['open'][curind] = data.price[r]
            df['high'][curind] = data['price'][r:i+1].max()
            df['low'][curind] = data['price'][r:i+1].min()
            df['close'][curind] = data.price[i]
            df['volume'][curind] = data['value'][r:i+1].sum()
            df['value'][curind] = valsum
            curind += 1
            valsum = 0
            r=i
    df['wavg'] = df.value / df.volume
    df = df.set_index('date')
    return df

def sampleRollingVolumeBars(data, interval=0, grain='H'):
    '''
    This function will sample bar data from raw df after a certain amount of volume.
    The volume is automatically determined from the rolling mean volume
    :param data: the raw df
    :param interval: the size of the rolling window the volume is computed
    default=0, means an expanding window will be used instead.
    :param grain: the granularity of the sampled bars approximated by the constant time bars.
    default='H', means that the bars will be sampled with the approx volume that is traded in an hour
    :return: a bar sampled df
    '''
    # first get the rolling or expanding volume for the grain time bars
    df_time = data.resample(grain).apply({'volume': 'sum', 'time': 'last'})
    # df_time = df_time.dropna()
    df_time = df_time[1:]  # remove the first mis values data point
    if interval == 0:
        avgvol = df_time['volume'].expanding(min_periods=1).mean()
    else:
        avgvol = df_time['volume'].rolling(window=interval).mean()
    avgvol = avgvol[~np.isnan(avgvol)]
    # now sample a bar from the raw data after each avgvol
    indx = range(0, 50000)  # guess max number of bars
    col = ['date', 'open', 'high', 'low', 'close', 'volume', 'value', 'wavg']
    df = pd.DataFrame(index=indx, columns=col)
    curind = 0
    volsum = 0  # volume counter
    r = 0  # previous bar index counter
    thr = 0  # tracking the index of avgvol
    for i in range(0, len(data)):
        volsum += data.volume[i]
        if thr < len(avgvol)-1:
            if data.index[i] > avgvol.index[thr+1]:
                thr += 1
        if volsum > avgvol[thr]:
            df['date'][curind] = data.index[i]  # the timestamp of the tick data is the date of the last tick
            df['open'][curind] = data.price[r]
            df['high'][curind] = data['price'][r:i+1].max()
            df['low'][curind] = data['price'][r:i+1].min()
            df['close'][curind] = data.price[i]
            df['volume'][curind] = volsum
            df['value'][curind] = data['value'][r:i+1].sum()
            curind += 1
            volsum = 0
            r = i
    df['wavg'] = df.value / df.volume
    df = df.set_index('date')
    return df

def sampleRollingEuroBars(data, interval=0, grain='H'):
    '''
    This function will sample bar data from raw df after a certain amount of value.
    The value is automatically determined from the rolling mean value
    :param data: the raw df
    :param interval: the size of the rolling window the mean value is computed over
    default=0, means an expanding window will be used instead.
    :param grain: the granularity of the sampled bars approximated by the constant time bars.
    default='H', means that the bars will be sampled with the approx value that is traded in an hour
    :return: a bar sampled df
    '''
    # first get the rolling or expanding volume for the grain time bars
    df_time = data.resample(grain).apply({'value': 'sum', 'time': 'last'})
    # df_time = df_time.dropna()
    df_time = df_time[1:]  # remove the first mis values data point
    if interval == 0:
        avgval = df_time['value'].expanding(min_periods=1).mean()
    else:
        avgval = df_time['value'].rolling(window=interval).mean()
    avgval = avgval[~np.isnan(avgval)]
    # now sample a bar from the raw data after each avgvol
    indx = range(0, 50000)  # guess max number of bars
    col = ['date', 'open', 'high', 'low', 'close', 'volume', 'value', 'wavg']
    df = pd.DataFrame(index=indx, columns=col)
    curind = 0
    valsum = 0  # volume counter
    r = 0  # previous bar index counter
    thr = 0  # tracking the index of avgval
    for i in range(0, len(data)):
        valsum += data.value[i]
        if thr < len(avgval)-1:
            if data.index[i] > avgval.index[thr+1]:
                thr += 1
        if valsum > avgval[thr]:
            df['date'][curind] = data.index[i]  # the timestamp of the tick data is the date of the last tick
            df['open'][curind] = data.price[r]
            df['high'][curind] = data['price'][r:i+1].max()
            df['low'][curind] = data['price'][r:i+1].min()
            df['close'][curind] = data.price[i]
            df['volume'][curind] = data['volume'][r:i+1].sum()
            df['value'][curind] = valsum
            curind += 1
            valsum = 0
            r = i
    df['wavg'] = df.value / df.volume
    df = df.set_index('date')
    return df

def getDailyVol(close, span0=100):
    '''
    Daily volatility estimated. A rolling exponential volatility is computed for the data series
    to get a good estimate for the threshold for which a label
    NOTE:
    There cannot be two samples with the same timestamp or the function does not work!
    Use: df0 = df0[~df0.index.duplicated(keep='last')] to remove the duplicates
    '''
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

def getTEvents(gRaw, h):
    '''
    A symmetric cumsum filter that detects upward and down ward deviations from a mean value.
    The raw time series data gRaw we want to filter and the threshold h value.
    '''
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos+diff.loc[i]), min(0, sNeg+diff.loc[i])
        if sNeg < -h:
            sNeg = 0;tEvents.append(i)
        elif sPos > h:
            sPos = 0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

def getTEventsMod(gRaw, h):
    '''
    Modified sample events. It takes a panda series of h threshold values
    '''
    tEvents,sPos,sNeg=[],0,0
    diff=gRaw.diff()
    for i in h.index[1:]:
        sPos,sNeg=max(0,sPos+diff.loc[i]),min(0,sNeg+diff.loc[i])
        if sNeg < -h[i]:
            sNeg=0;tEvents.append(i)
        elif sPos > h[i]:
            sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

def applyPtSlOnT1 (close,events,ptSl):
    '''
    A function with four arguements:
     close: A pandas series of prices.
     events: A pandas dataframe, with columns,
        ◦ t1: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
        ◦ trgt: The unit width of the horizontal barriers.
     ptSl: A list of two non-negative float values:
        ◦ ptSl[0]:The factor that multiplies trgt to set the width of the upper barrier. If 0, there will not be an upper barrier.
        ◦ ptSl[1]:The factor that multiplies trgt to set the width of the lower barrier. If 0, there will not be a lower barrier.
     molecule: A list with the subset of event indices that will be processed by a single thread.
    Its use will become clear later on in the chapter.
    '''
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    #events_=events.loc[molecule]
    events_=events
    out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0:
        pt=ptSl[0]*events_['trgt']
    else:
        pt=pd.Series(index=events.index) #NaNs
    if ptSl[1]>0:
        sl=-ptSl[1]*events_['trgt']
    else:
        sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking
    return out

def getEvents(close,tEvents,ptSl,trgt,minRet,t1=False,side=None):
    '''
    A functin that gets the events of the barier touch.
     close: A pandas series of prices.
     tEvents:The pandas time index containing the time stamps that will seed every triple barrier. These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
     ptSl: A non-negative float that sets the width of the two barriers. A 0 value means that the respective horizontal barrier (profit taking and/or stop loss) will be disabled.
     t1: A pandas series with the timestamps of the vertical barriers. We pass a False when we want to disable vertical barriers.
     trgt: A pandas series of targets, expressed in terms of absolute returns.
     minRet: The minimum target return required for running a triple barrier search.
     numThreads: The number of threads concurrently used by the function.
    '''
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False: t1=pd.Series(pd.NaT,index=tEvents)
    #3) form events object, apply stop loss on t1
    if side is None:
        side_,ptSl_ = pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
    else:
        side_,ptSl_ = side.loc[trgt.index],ptSl[:2]
    events=pd.concat({'t1':t1,'trgt':trgt,'side':side_},axis=1).dropna(subset=['trgt'])
    # multi process function
    #df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),numThreads=numThreads,close=close,events=events,ptSl=[ptSl,ptSl])
    #no mp modification
    df0=applyPtSlOnT1(close=close,events=events,ptSl=ptSl_)
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    if side is None: events=events.drop('side',axis=1)
    return events

def getVertT1(close,tEvents,numDays):
    '''
    Method to get the timestamps t1 of the verticle barrier. Holding period = numDays
    '''
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]) # NaNs at end
    return t1

def getBins(events,close):
    '''
     ret: The return realized at the time of the first touched barrier.
     bin:The label,{−1,0,1}, as a function of the sign of the outcome.
    Thefunction can be easily adjusted to label as 0 those events when the vertical barrier was touched first, which we leave as an exercise.
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_: out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_: out.loc[out['ret']<=0,'bin']=0 # meta_labeling
    return out

def mpNumCoEvents(closeIdx,t1):
    '''
    Number of concurrent labels. The number of labels that are computed from the return at t.
    closeIdx is the index of the close sampled data.
    t1 is the series of barrier touches: index, time (of first touch)
    '''
    t1=t1.ﬁllna(closeIdx[-1]) # unclosed events still must impact other weights
    # count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.iteritems():
        count.loc[tIn:tOut]+=1
    return count

def mpSampleTW(t1,numCoEvents):
    '''
    Estimating the average uniqueness of a label.
    '''
    # Derive average uniqueness over the event's lifespan
    wght=pd.Series(index=t1.index)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght

def mpSampleW(t1,numCoEvents,close, returns=False):
    '''
    WEIGHT to a sample for training of a ML algo.
    The weight is a function of the uniqueness and the returns.
    The results W then need to scale so that all w sum to num_samples:
    W = w*num_sample*(1/sum_w):
    W = w
    W *= W.shape[0]/W.sum()
    '''
    # Derive sample weight by return attribution
    # ret = data.close.astype(np.float64).apply(np.log).diff()
    ret = np.log(close).diff()
    wght=pd.Series(index=t1.index)
    rets = pd.Series(index=t1.index)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
        rets.loc[tIn]=(ret.loc[tIn:tOut]).sum()
    if returns:
        return wght.abs(), rets
    else:
        return wght.abs()

def getTimeDecay(tW,clfLastW=1.):
    '''
    Time-Decay of weights by uniqueness.
    Returns the decay factor for each sample. Can then be used to multiply by the weights W.
    '''
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0:
        slope=(1.-clfLastW)/clfW.iloc[-1]
    else:
        slope=1./((clfLastW+1)*clfW.iloc[-1])
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    #print(const,slope)
    return clfW

import statsmodels.api as sm1
#--------------------------------------------------
def tValLinR(close):
    # tValue from a linear trend
    x=np.ones((close.shape[0],2))
    x[:,1]=np.arange(close.shape[0])
    ols=sm1.OLS(close,x).ﬁt()
    return ols.tvalues[1]

def getBinsFromTrend(molecule,close,span):
    ''' Derive labels from the sign of t-value of linear trend
    Output includes:
    - t1: End time for the identiﬁed trend
    - tVal: t-value associated with the estimated trend coefﬁcient
    - bin: Sign of the trend
    '''
    out=pd.DataFrame(index=molecule,columns=['t0','t1','tVal','bin'])
    hrzns=range(*span)
    for dt0 in molecule:
        df0=pd.Series()
        iloc0=close.index.get_loc(dt0)
        if iloc0+max(hrzns)>close.shape[0]:continue
        for hrzn in hrzns:
            dt1=close.index[iloc0+hrzn-1]
            df1=close.loc[dt0:dt1]
            df0.loc[dt1]=tValLinR(df1.values)
        dt1=df0.replace([-np.inf,np.inf,np.nan],0).abs().idxmax()
        out.loc[dt0,['t0','t1','tVal','bin']]=dt1,df0.index[-1],df0[dt1], np.sign(df0[dt1]) # prevent leakage
    out['t0']=pd.to_datetime(out['t0'])
    out['t1']=pd.to_datetime(out['t1'])
    out['bin']=pd.to_numeric(out['bin'],downcast='signed')
    return out.dropna(subset=['bin'])