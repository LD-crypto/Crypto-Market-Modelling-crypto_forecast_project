from src.features import mass_features as mf
from src.features import build_features as bf
import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse
from statsmodels.tsa.stattools import adfuller
import datetime

'''
The aim of this script is to create a dataframe from each sampled datasets of stationary features.
Each feature will be fractionally differentiated to the lowest value of d for which the feature is stationary
The final dataframes will be saved into the processsed data folder. The interim un-differentiated features
and the d-values will be saved into the interim data folder.
'''

def getDValues(df_list, d_list=None):
    if d_list is None:
        d_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    # join the data into one dataframe
    df = df_list[0]
    if len(df_list) != 1:
        for i in range(1, len(df_list)):
            df = df.append(df_list[i])
    # find the non-stationary features
    df_p = pd.DataFrame(index=['p-value'], columns=df.columns)
    non_st = []
    for col in df.columns:
        df_p.loc['p-value', col] = adfuller(df[col].replace([np.inf, -np.inf], np.nan).dropna(), maxlag=1)[1]
    non_st = df_p[df_p > 0.001]
    a = non_st.dropna(axis=1)
    non_stationary_features = list(a.columns)
    st = df_p[df_p < 0.001]
    b = st.dropna(axis=1)
    stationary_features = list(b.columns)
    # frac diff for each d and compute the p-values of adf statistic
    d_values = pd.DataFrame(index=non_stationary_features, columns=d_list)
    for d in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        # fracdiff the series
        df = df_list[0]
        df1 = bf.fracDiff_FFD(df[non_stationary_features], d, thres=0.001)
        if len(df_list) != 1:
            for i in range(1, len(df_list)):
                df = df_list[i]
                df2 = bf.fracDiff_FFD(df[non_stationary_features], d, thres=0.001)
                df1 = df1.append(df2)
        for ft in non_stationary_features:
            # perform adfuller test on each fracdiff-feature
            adf = adfuller(df1[ft].replace([np.inf, -np.inf], np.nan).dropna(), maxlag=1, regression='c', autolag=None)
            d_values.loc[ft][d] = adf[1]  # p-value
    return d_values

def makeStationary(df_list, d_values):
    ds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    features = list([])
    for i in d_values.index:
        n = 0
        while d_values.loc[i][ds[n]] > 0.0001:
            if n == 9:
                break
            n += 1
        features.append((i, ds[n]))
    # create data frame with already stationary features
    df = df_list[0]
    for i in range(1,len(df_list)):
        df = df.append(df_list[i])
    stationary_df = pd.DataFrame(df.drop(columns=d_values.index.values))
    # for each feature fracDiff it and add it to the df
    for ft in features:
        dft = pd.DataFrame(df_list[0][ft[0]])
        fDiff_df = bf.fracDiff_FFD(dft, ft[1], thres=0.001)
        for i in range(1,len(df_list)):
            # frac diff each market segment and then join them
            dfi = pd.DataFrame(df_list[i][ft[0]])
            fDiff_dfi = bf.fracDiff_FFD(dfi, ft[1], thres=0.001)
            fDiff_df = fDiff_df.append(fDiff_dfi)
        stationary_df[ft[0]] = fDiff_df[ft[0]]
    return stationary_df

def main(folder_name):
    '''
    Given a folder name. This function will run through each market in the folder, create mass features, find the
    smallest d for which a feature is stationary then create a fully stationary data set.
    '''
    print('SCRIPT STARTED AT: '+str(datetime.datetime.now()))
    # check if the folder_name\exists
    folder_path = str(Path().absolute().parent.parent)+'/data/interim/'+folder_name
    if not os.path.exists(folder_path + '/raw'):
        raise Exception('{0} or {0}/raw does not exist in /data/interim.'.format(folder_name))

    for market in os.listdir(folder_path+'/raw'):
        # markets are the names of the market folders containing the bar datafiles
        if os.path.exists(str(Path().absolute().parent.parent)+'/data/processed/'+folder_name+'/'+market+'/'+market+'_stationary.pkl'):
            continue
        print('Starting {}...'.format(market)+' at time: '+str(datetime.datetime.now()))
        # CREATE+SAVE MASS FEATURES
        if not os.path.exists(folder_path + '/features/'+market):
            os.makedirs(folder_path + '/features/'+market)
            df = []
            for datafile in os.listdir(folder_path + '/raw/' + market):
                dataf = mf.getMassFeatures(pd.read_pickle(folder_path + '/raw/' + market + '/' + datafile))
                dataf.to_pickle(folder_path + '/features/' + market + '/features_' + datafile)
                df.append(dataf)
            print('Mass features created and saved.')
        else:
            df = []
            for datafile in os.listdir(folder_path + '/features/' + market):
                df.append(pd.read_pickle(folder_path + '/features/' + market + '/' + datafile))

        #FRACDIFF+TEST+SAVE D_VALUES
        print('Fractionally differentiating features and generating D_VALUES....')
        if not os.path.exists(folder_path + '/features/'+market+'/'+market+'_d_values.pkl'):
            d_values = getDValues(df)
            d_values.to_pickle(folder_path + '/features/'+market+'/'+market+'_d_values.pkl')
            print('D_VALUES successfully generated and saved in: '+folder_path+'/features/'+market)
        else:
            d_values = pd.read_pickle(folder_path + '/features/'+market+'/'+market+'_d_values.pkl')

        # CREATE+SAVE FINAL STATIONARY SET
        print('Creating stationary data set of ' + str(market) + '...')
        stationary_df = makeStationary(df, d_values)
        if not os.path.exists(str(Path().absolute().parent.parent)+'/data/processed/'+folder_name+'/'+market):
            os.makedirs(str(Path().absolute().parent.parent)+'/data/processed/'+folder_name+'/'+market)
        stationary_df.to_pickle(str(Path().absolute().parent.parent)+'/data/processed/'+folder_name+'/'+market+'/'+market+'_stationary.pkl')
        print(market+' dataset successfully generated and saved in: ' + folder_path + '/features/' + market)
    print('SCRIPT COMPLETED AT: '+str(datetime.datetime.now()))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--folder',
        help='name of folder the data is stored in',
        type=str,
        default='sample_method')

    args = parser.parse_args()

    # execute
    main(args.folder)