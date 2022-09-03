'''
A mass feature constructor function taking the raw bar data as input and outputting the feature df
'''
import numpy as np
import pandas as pd
from stockstats import StockDataFrame
import os

def getPeriodHigh(series, nperiods):
    # function marks the last highest high in n periods
    highest = series.rolling(window=nperiods).max()
    return highest

def getPeriodLow(series, nperiods):
    # function marks the last lowest low in n periods
    lowest = series.rolling(window=nperiods).min()
    return lowest

def CCI(data, ndays):
    # momentum oscillators
    # Commodity Channel Index (CCI)
    TP = (data['high'] + data['low'] + data['close'])/3
    CCI = pd.Series((TP - TP.rolling(window=ndays).mean()) / (0.015 * TP.rolling(window=ndays).std()))
    return CCI

def getTR(dataframe):
    '''
    This function computes the true rate of a dataframe at each position
    '''
    df = pd.DataFrame(index=dataframe.index)
    df['ATR1'] = abs(dataframe['high'] - dataframe['low'])
    df['ATR2'] = abs(dataframe['high'] - dataframe['close'].shift())
    df['ATR3'] = abs(dataframe['low'] - dataframe['close'].shift())
    df['tr'] = df[['ATR1', 'ATR2', 'ATR3']].max(axis=1)
    return df['tr']

def getTRlog(dataframe):
    '''
    This function computes the true rate of a dataframe at each position
    '''
    df = pd.DataFrame(index=dataframe.index)
    df['ATR1'] = abs(dataframe['loghigh'] - dataframe['loglow'])
    df['ATR2'] = abs(dataframe['loghigh'] - dataframe['logclose'].shift())
    df['ATR3'] = abs(dataframe['loglow'] - dataframe['logclose'].shift())
    df['logtr'] = df[['ATR1', 'ATR2', 'ATR3']].max(axis=1)
    return df['logtr']

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

def getMassFeatures(data):
    d1 = data
    # The returns series
    d1['ret'] = d1.close.pct_change()
    # First the simple log and returns series for each raw feature will be created
    d1['logopen'] = d1.open.astype(np.float64).apply(np.log)
    d1['loghigh'] = d1.high.astype(np.float64).apply(np.log)
    d1['loglow'] = d1.low.astype(np.float64).apply(np.log)
    d1['logclose'] = d1.close.astype(np.float64).apply(np.log)
    d1['logwavg'] = d1.wavg.astype(np.float64).apply(np.log)
    d1['logvolume'] = d1.volume.astype(np.float64).apply(np.log)
    d1['logvalue'] = d1.value.astype(np.float64).apply(np.log)
    # ohlc relative diff
    d1['oh'] = d1.logopen-d1.loghigh
    d1['ol'] = d1.logopen - d1.loglow
    d1['hl'] = d1.loghigh - d1.loglow
    d1['hc'] = d1.loghigh - d1.logclose
    d1['lc'] = d1.loglow - d1.logclose
    # approx first differentials of close and wavg
    d1['logret'] = d1['logclose'].diff(1)
    d1['5_1dclose'] = d1['logclose'].diff(5)
    d1['10_1dclose'] = d1['logclose'].diff(10)
    d1['20_1dclose'] = d1['logclose'].diff(20)
    d1['50_1dclose'] = d1['logclose'].diff(50)
    d1['100_1dclose'] = d1['logclose'].diff(100)
    d1['200_1dclose'] = d1['logclose'].diff(200)
    d1['500_1dclose'] = d1['logclose'].diff(500)
    d1['1_1dwavg'] = d1['logwavg'].diff(1)
    d1['5_1dwavg'] = d1['logwavg'].diff(5)
    d1['10_1dwavg'] = d1['logwavg'].diff(10)
    d1['20_1dwavg'] = d1['logwavg'].diff(20)
    d1['50_1dwavg'] = d1['logwavg'].diff(50)
    d1['100_1dwavg'] = d1['logwavg'].diff(100)
    d1['200_1dwavg'] = d1['logwavg'].diff(200)
    d1['500_1dwavg'] = d1['logwavg'].diff(500)
    # approx second differentials
    d1['1_2dclose'] = d1['logret'].diff(1)
    d1['5_2dclose'] = d1['5_1dclose'].diff(5)
    d1['10_2dclose'] = d1['10_1dclose'].diff(10)
    d1['20_2dclose'] = d1['20_1dclose'].diff(20)
    d1['50_2dclose'] = d1['50_1dclose'].diff(50)
    d1['100_2dclose'] = d1['100_1dclose'].diff(100)
    d1['200_2dclose'] = d1['200_1dclose'].diff(200)
    d1['500_2dclose'] = d1['500_1dclose'].diff(500)
    d1['1_2dwavg'] = d1['1_1dwavg'].diff(1)
    d1['5_2dwavg'] = d1['5_1dwavg'].diff(5)
    d1['10_2dwavg'] = d1['10_1dwavg'].diff(10)
    d1['20_2dwavg'] = d1['20_1dwavg'].diff(20)
    d1['50_2dwavg'] = d1['50_1dwavg'].diff(50)
    d1['100_2dwavg'] = d1['100_1dwavg'].diff(100)
    d1['200_2dwavg'] = d1['200_1dwavg'].diff(200)
    d1['500_2dwavg'] = d1['500_1dwavg'].diff(500)
    # different periods log volume approx first differential; 1,5,10,20,50,100,200,500
    d1['1_1dvolume'] = d1.logvolume.diff(1)
    d1['5_1dvolume'] = d1.logvolume.diff(5)
    d1['10_1dvolume'] = d1.logvolume.diff(10)
    d1['20_1dvolume'] = d1.logvolume.diff(20)
    d1['50_1dvolume'] = d1.logvolume.diff(50)
    d1['100_1dvolume'] = d1.logvolume.diff(100)
    d1['200_1dvolume'] = d1.logvolume.diff(200)
    d1['500_1dvolume'] = d1.logvolume.diff(500)
    # different periods log value approx first differential; 1,5,10,20,50,100,200,500
    d1['1_1dvalue'] = d1.logvalue.diff(1)
    d1['5_1dvalue'] = d1.logvalue.diff(5)
    d1['10_1dvalue'] = d1.logvalue.diff(10)
    d1['20_1dvalue'] = d1.logvalue.diff(20)
    d1['50_1dvalue'] = d1.logvalue.diff(50)
    d1['100_1dvalue'] = d1.logvalue.diff(100)
    d1['200_1dvalue'] = d1.logvalue.diff(200)
    d1['500_1dvalue'] = d1.logvalue.diff(500)
    # Simple moving averages/std's
    # 5,10,20,50,100,200,500,1000,2400
    d1['sma5'] = d1['close'].rolling(window=5).mean()
    d1['smah5'] = d1['high'].rolling(window=5).mean()
    d1['smal5'] = d1['low'].rolling(window=5).mean()
    d1['sma10'] = d1['close'].rolling(window=10).mean()
    d1['smah10'] = d1['high'].rolling(window=10).mean()
    d1['smal10'] = d1['low'].rolling(window=10).mean()
    d1['sma20'] = d1['close'].rolling(window=20).mean()
    d1['smah20'] = d1['high'].rolling(window=20).mean()
    d1['smal20'] = d1['low'].rolling(window=20).mean()
    d1['sma50'] = d1['close'].rolling(window=50).mean()
    d1['smah50'] = d1['high'].rolling(window=50).mean()
    d1['smal50'] = d1['low'].rolling(window=50).mean()
    d1['sma100'] = d1['close'].rolling(window=100).mean()
    d1['smah100'] = d1['high'].rolling(window=100).mean()
    d1['smal100'] = d1['low'].rolling(window=100).mean()
    d1['sma200'] = d1['close'].rolling(window=200).mean()
    d1['smah200'] = d1['high'].rolling(window=200).mean()
    d1['smal200'] = d1['low'].rolling(window=200).mean()
    d1['sma500'] = d1['close'].rolling(window=500).mean()
    d1['smah500'] = d1['high'].rolling(window=500).mean()
    d1['smal500'] = d1['low'].rolling(window=500).mean()
    d1['sma1000'] = d1['close'].rolling(window=1000).mean()
    d1['smah1000'] = d1['high'].rolling(window=1000).mean()
    d1['smal1000'] = d1['low'].rolling(window=1000).mean()
    d1['sma2400'] = d1['close'].rolling(window=2400).mean()
    d1['smah2400'] = d1['high'].rolling(window=2400).mean()
    d1['smal2400'] = d1['low'].rolling(window=2400).mean()
    # Simple moving averages of log price
    # 5,10,20,50,100,200,500,1000,2400
    d1['logsma5'] = d1['logclose'].rolling(window=5).mean()
    d1['logsmah5'] = d1['loghigh'].rolling(window=5).mean()
    d1['logsmal5'] = d1['loglow'].rolling(window=5).mean()
    d1['logsma10'] = d1['logclose'].rolling(window=10).mean()
    d1['logsmah10'] = d1['loghigh'].rolling(window=10).mean()
    d1['logsmal10'] = d1['loglow'].rolling(window=10).mean()
    d1['logsma20'] = d1['logclose'].rolling(window=20).mean()
    d1['logsmah20'] = d1['loghigh'].rolling(window=20).mean()
    d1['logsmal20'] = d1['loglow'].rolling(window=20).mean()
    d1['logsma50'] = d1['logclose'].rolling(window=50).mean()
    d1['logsmah50'] = d1['loghigh'].rolling(window=50).mean()
    d1['logsmal50'] = d1['loglow'].rolling(window=50).mean()
    d1['logsma100'] = d1['logclose'].rolling(window=100).mean()
    d1['logsmah100'] = d1['loghigh'].rolling(window=100).mean()
    d1['logsmal100'] = d1['loglow'].rolling(window=100).mean()
    d1['logsma200'] = d1['logclose'].rolling(window=200).mean()
    d1['logsmah200'] = d1['loghigh'].rolling(window=200).mean()
    d1['logsmal200'] = d1['loglow'].rolling(window=200).mean()
    d1['logsma500'] = d1['logclose'].rolling(window=500).mean()
    d1['logsmah500'] = d1['loghigh'].rolling(window=500).mean()
    d1['logsmal500'] = d1['loglow'].rolling(window=500).mean()
    d1['logsma1000'] = d1['logclose'].rolling(window=1000).mean()
    d1['logsmah1000'] = d1['loghigh'].rolling(window=1000).mean()
    d1['logsmal1000'] = d1['loglow'].rolling(window=1000).mean()
    d1['logsma2400'] = d1['logclose'].rolling(window=2400).mean()
    d1['logsmah2400'] = d1['loghigh'].rolling(window=2400).mean()
    d1['logsmal2400'] = d1['loglow'].rolling(window=2400).mean()
    # rolling standard deviation of log returns
    d1['std5'] = d1['logret'].rolling(window=5).std()
    d1['std10'] = d1['logret'].rolling(window=10).std()
    d1['std20'] = d1['logret'].rolling(window=20).std()
    d1['std50'] = d1['logret'].rolling(window=50).std()
    d1['std100'] = d1['logret'].rolling(window=100).std()
    d1['std200'] = d1['logret'].rolling(window=200).std()
    d1['std500'] = d1['logret'].rolling(window=500).std()
    d1['std1000'] = d1['logret'].rolling(window=1000).std()
    d1['std2400'] = d1['logret'].rolling(window=2400).std()
    # ATR 5,10,20,50,100
    d1['tr'] = getTR(d1)
    d1['atr5'] = d1['tr'].rolling(window=5).mean()
    d1['atr10'] = d1['tr'].rolling(window=10).mean()
    d1['atr20'] = d1['tr'].rolling(window=20).mean()
    d1['atr50'] = d1['tr'].rolling(window=50).mean()
    d1['atr100'] = d1['tr'].rolling(window=100).mean()
    # logATR 5,10,20,50,100
    d1['logtr'] = getTRlog(d1)
    d1['logatr5'] = d1['logtr'].rolling(window=5).mean()
    d1['logatr10'] = d1['logtr'].rolling(window=10).mean()
    d1['logatr20'] = d1['logtr'].rolling(window=20).mean()
    d1['logatr50'] = d1['logtr'].rolling(window=50).mean()
    d1['logatr100'] = d1['logtr'].rolling(window=100).mean()
    # daily volatility of close,high,low
    d1['dvol'] = getDailyVol(d1.close)
    d1['dvolh'] = getDailyVol(d1.high)
    d1['dvoll'] = getDailyVol(d1.low)
    d1['dvolwavg'] = getDailyVol(d1.wavg)
    # CCI 2,5,10,20,50,100,200,500,1000
    d1['cci2'] = CCI(d1, 2)
    d1['cci5'] = CCI(d1, 5)
    d1['cci10'] = CCI(d1, 10)
    d1['cci20'] = CCI(d1, 20)
    d1['cci50'] = CCI(d1, 50)
    d1['cci100'] = CCI(d1, 100)
    d1['cci200'] = CCI(d1, 200)
    d1['cci500'] = CCI(d1, 500)
    d1['cci1000'] = CCI(d1, 1000)
    # RSI
    d1_s = StockDataFrame.retype(d1)
    d1['rsi2'] = d1_s.loc[:]['rsi_2']
    d1['rsi5'] = d1_s.loc[:]['rsi_5']
    d1['rsi10'] = d1_s.loc[:]['rsi_10']
    d1['rsi20'] = d1_s.loc[:]['rsi_20']
    d1['rsi50'] = d1_s.loc[:]['rsi_50']
    d1['rsi100'] = d1_s.loc[:]['rsi_100']
    d1['rsi200'] = d1_s.loc[:]['rsi_200']
    d1['rsi500'] = d1_s.loc[:]['rsi_500']
    # Williams R
    d1['wr2'] = d1_s.loc[:]['wr_2']
    d1['wr5'] = d1_s.loc[:]['wr_5']
    d1['wr10'] = d1_s.loc[:]['wr_10']
    d1['wr20'] = d1_s.loc[:]['wr_20']
    d1['wr50'] = d1_s.loc[:]['wr_50']
    d1['wr100'] = d1_s.loc[:]['wr_100']
    d1['wr200'] = d1_s.loc[:]['wr_200']
    d1['wr500'] = d1_s.loc[:]['wr_500']
    # difference between SMAs
    # DMAX-Y
    # 5,10,20,50,100,200,500,1000,2400
    # closing moving averages difference
    d1['dma1-5'] = d1.close - d1.sma5
    d1['dma1-10'] = d1.close - d1.sma10
    d1['dma5-10'] = d1.sma5 - d1.sma10
    d1['dma5-20'] = d1.sma5 - d1.sma20
    d1['dma10-20'] = d1.sma10 - d1.sma20
    d1['dma10-50'] = d1.sma10 - d1.sma50
    d1['dma20-50'] = d1.sma20 - d1.sma50
    d1['dma20-100'] = d1.sma20 - d1.sma100
    d1['dma50-100'] = d1.sma50 - d1.sma100
    d1['dma50-200'] = d1.sma50 - d1.sma200
    d1['dma100-200'] = d1.sma100 - d1.sma200
    d1['dma100-500'] = d1.sma100 - d1.sma500
    d1['dma200-500'] = d1.sma200 - d1.sma500
    d1['dma200-1000'] = d1.sma200 - d1.sma1000
    d1['dma500-1000'] = d1.sma500 - d1.sma1000
    d1['dma500-2400'] = d1.sma500 - d1.sma2400
    # high price dma
    d1['dmah1-5'] = d1.high - d1.smah5
    d1['dmah1-10'] = d1.high - d1.smah10
    d1['dmah5-10'] = d1.smah5 - d1.smah10
    d1['dmah5-20'] = d1.smah5 - d1.smah20
    d1['dmah10-20'] = d1.smah10 - d1.smah20
    d1['dmah10-50'] = d1.smah10 - d1.smah50
    d1['dmah20-50'] = d1.smah20 - d1.smah50
    d1['dmah20-100'] = d1.smah20 - d1.smah100
    d1['dmah50-100'] = d1.smah50 - d1.smah100
    d1['dmah50-200'] = d1.smah50 - d1.smah200
    d1['dmah100-200'] = d1.smah100 - d1.smah200
    d1['dmah100-500'] = d1.smah100 - d1.smah500
    d1['dmah200-500'] = d1.smah200 - d1.smah500
    d1['dmah200-1000'] = d1.smah200 - d1.smah1000
    d1['dmah500-1000'] = d1.smah500 - d1.smah1000
    d1['dmah500-2400'] = d1.smah500 - d1.smah2400
    # low price dma
    d1['dmal1-5'] = d1.low - d1.smal5
    d1['dmal1-10'] = d1.low - d1.smal10
    d1['dmal5-10'] = d1.smal5 - d1.smal10
    d1['dmal5-20'] = d1.smal5 - d1.smal20
    d1['dmal10-20'] = d1.smal10 - d1.smal20
    d1['dmal10-50'] = d1.smal10 - d1.smal50
    d1['dmal20-50'] = d1.smal20 - d1.smal50
    d1['dmal20-100'] = d1.smal20 - d1.smal100
    d1['dmal50-100'] = d1.smal50 - d1.smal100
    d1['dmal50-200'] = d1.smal50 - d1.smal200
    d1['dmal100-200'] = d1.smal100 - d1.smal200
    d1['dmal100-500'] = d1.smal100 - d1.smal500
    d1['dmal200-500'] = d1.smal200 - d1.smal500
    d1['dmal200-1000'] = d1.smal200 - d1.smal1000
    d1['dmal500-1000'] = d1.smal500 - d1.smal1000
    d1['dmal500-2400'] = d1.smal500 - d1.smal2400
    # difference between logSMAs
    # logDMAX-Y
    # 5,10,20,50,100,200,500,1000,2400
    # closing moving averages difference
    d1['logdma1-5'] = d1.logclose - d1.logsma5
    d1['logdma1-10'] = d1.logclose - d1.logsma10
    d1['logdma5-10'] = d1.logsma5 - d1.logsma10
    d1['logdma5-20'] = d1.logsma5 - d1.logsma20
    d1['logdma10-20'] = d1.logsma10 - d1.logsma20
    d1['logdma10-50'] = d1.logsma10 - d1.logsma50
    d1['logdma20-50'] = d1.logsma20 - d1.logsma50
    d1['logdma20-100'] = d1.logsma20 - d1.logsma100
    d1['logdma50-100'] = d1.logsma50 - d1.logsma100
    d1['logdma50-200'] = d1.logsma50 - d1.logsma200
    d1['logdma100-200'] = d1.logsma100 - d1.logsma200
    d1['logdma100-500'] = d1.logsma100 - d1.logsma500
    d1['logdma200-500'] = d1.logsma200 - d1.logsma500
    d1['logdma200-1000'] = d1.logsma200 - d1.logsma1000
    d1['logdma500-1000'] = d1.logsma500 - d1.logsma1000
    d1['logdma500-2400'] = d1.logsma500 - d1.logsma2400
    # high price dma
    d1['logdmah1-5'] = d1.loghigh - d1.logsmah5
    d1['logdmah1-10'] = d1.loghigh - d1.logsmah10
    d1['logdmah5-10'] = d1.logsmah5 - d1.logsmah10
    d1['logdmah5-20'] = d1.logsmah5 - d1.logsmah20
    d1['logdmah10-20'] = d1.logsmah10 - d1.logsmah20
    d1['logdmah10-50'] = d1.logsmah10 - d1.logsmah50
    d1['logdmah20-50'] = d1.logsmah20 - d1.logsmah50
    d1['logdmah20-100'] = d1.logsmah20 - d1.logsmah100
    d1['logdmah50-100'] = d1.logsmah50 - d1.logsmah100
    d1['logdmah50-200'] = d1.logsmah50 - d1.logsmah200
    d1['logdmah100-200'] = d1.logsmah100 - d1.logsmah200
    d1['logdmah100-500'] = d1.logsmah100 - d1.logsmah500
    d1['logdmah200-500'] = d1.logsmah200 - d1.logsmah500
    d1['logdmah200-1000'] = d1.logsmah200 - d1.logsmah1000
    d1['logdmah500-1000'] = d1.logsmah500 - d1.logsmah1000
    d1['logdmah500-2400'] = d1.logsmah500 - d1.logsmah2400
    # low price mda
    d1['logdmal1-5'] = d1.loglow - d1.logsmal5
    d1['logdmal1-10'] = d1.loglow - d1.logsmal10
    d1['logdmal5-10'] = d1.logsmal5 - d1.logsmal10
    d1['logdmal5-20'] = d1.logsmal5 - d1.logsmal20
    d1['logdmal10-20'] = d1.logsmal10 - d1.logsmal20
    d1['logdmal10-50'] = d1.logsmal10 - d1.logsmal50
    d1['logdmal20-50'] = d1.logsmal20 - d1.logsmal50
    d1['logdmal20-100'] = d1.logsmal20 - d1.logsmal100
    d1['logdmal50-100'] = d1.logsmal50 - d1.logsmal100
    d1['logdmal50-200'] = d1.logsmal50 - d1.logsmal200
    d1['logdmal100-200'] = d1.logsmal100 - d1.logsmal200
    d1['logdmal100-500'] = d1.logsmal100 - d1.logsmal500
    d1['logdmal200-500'] = d1.logsmal200 - d1.logsmal500
    d1['logdmal200-1000'] = d1.logsmal200 - d1.logsmal1000
    d1['logdmal500-1000'] = d1.logsmal500 - d1.logsmal1000
    d1['logdmal500-2400'] = d1.logsmal500 - d1.logsmal2400
    # difference between current close and last period high
    d1['lasthigh5'] = getPeriodHigh(d1.loghigh, 5) - d1.logclose
    d1['lasthigh10'] = getPeriodHigh(d1.loghigh, 10) - d1.logclose
    d1['lasthigh20'] = getPeriodHigh(d1.loghigh, 20) - d1.logclose
    d1['lasthigh50'] = getPeriodHigh(d1.loghigh, 50) - d1.logclose
    d1['lasthigh100'] = getPeriodHigh(d1.loghigh, 100) - d1.logclose
    d1['lasthigh200'] = getPeriodHigh(d1.loghigh, 200) - d1.logclose
    d1['lasthigh500'] = getPeriodHigh(d1.loghigh, 500) - d1.logclose
    d1['lasthigh1000'] = getPeriodHigh(d1.loghigh, 1000) - d1.logclose
    d1['lasthigh2400'] = getPeriodHigh(d1.loghigh, 2400) - d1.logclose
    # difference between current close and last period low
    d1['lastlow5'] = getPeriodLow(d1.loglow, 5) - d1.logclose
    d1['lastlow10'] = getPeriodLow(d1.loglow, 10) - d1.logclose
    d1['lastlow20'] = getPeriodLow(d1.loglow, 20) - d1.logclose
    d1['lastlow50'] = getPeriodLow(d1.loglow, 50) - d1.logclose
    d1['lastlow100'] = getPeriodLow(d1.loglow, 100) - d1.logclose
    d1['lastlow200'] = getPeriodLow(d1.loglow, 200) - d1.logclose
    d1['lastlow500'] = getPeriodLow(d1.loglow, 500) - d1.logclose
    d1['lastlow1000'] = getPeriodLow(d1.loglow, 1000) - d1.logclose
    d1['lastlow2400'] = getPeriodLow(d1.loglow, 2400) - d1.logclose
    return d1





