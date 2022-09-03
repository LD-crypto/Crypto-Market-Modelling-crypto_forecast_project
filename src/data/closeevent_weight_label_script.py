from src.data import build_dataset as bd
import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse
import datetime

'''
The aim of this script is to...
'''

def main(folder_name, e_criteria, period, target, decay):
    '''
    Given a folder name. This function will run through each market in the folder, create mass features, find the
    smallest d for which a feature is stationary then create a fully stationary data set.
    :param folder_name:
    :param e_criteria: a multiple of the daily volatility used to define an event
    :param period: number of days to use for vertical barrier
    :param target: multiple of the daily volatility used for horizontal barriers
    :param decay: % weight reduction per year
    '''
    print('SCRIPT STARTED AT: '+str(datetime.datetime.now()))
    # check if the folder_name\exists
    folder_path = str(Path().absolute().parent.parent)+'/data/raw/'+folder_name
    if not os.path.exists(folder_path):
        raise Exception('{0} does not exist in /data/raw.'.format(folder_name))

    for dataset in os.listdir(folder_path):
        # markets are the names of the market folders containing the bar datafiles
        # if os.path.exists(str(Path().absolute().parent.parent)+'/data/processed/'+folder_name+'/'+market+'/'+market+'_stationary.pkl'):
         #   continue
        print('Starting {}...'.format(dataset)+' at time: '+str(datetime.datetime.now()))
        # SAMPLE EVENTS + GET LABELS
        data = pd.read_pickle(folder_path + '/' + dataset)
        dVol = bd.getDailyVol(data.close)
        tEvents = bd.getTEventsMod(np.log(data.close), dVol * e_criteria)
        t1 = bd.getVertT1(data.close, tEvents, period)
        events = bd.getEvents(data.close, tEvents, 1, dVol * target, dVol.min() * target, t1)
        out = bd.getBins(events, data.close)

        # CALCULATE WEIGHTS FOR THE SAMPLES
        numCoEvents = bd.mpNumCoEvents(data.index, events.t1)
        out['tW'] = bd.mpSampleTW(events.t1, numCoEvents)
        out['W'] = bd.mpSampleW(events.t1, numCoEvents, data.close)
        out['W'] *= out.shape[0] / out['W'].sum()
        td = data.index[-1] - data.index[0]
        num_years = td.days / 365
        decay_param = 1 - (decay / 100) * num_years
        dW = bd.getTimeDecay(out['tW'], clfLastW=decay_param)
        out['dW'] = out['W'] * dW

        # SAVE THE OUTPUTS
        fin_folder = str(Path().absolute().parent.parent)+'/data/processed/'+folder_name+'/'+str(e_criteria)+'_'+str(period)+'_'+str(target)+'_'+str(decay)
        if not os.path.exists(fin_folder):
            os.makedirs(fin_folder)
        out.to_pickle(fin_folder+'/output_'+dataset)
        print('Output of '+str(dataset)+' successfully generated and saved in: ' + fin_folder)
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

    parser.add_argument(
        '--e_criteria',
        help='A float multiple of the std of price change that defines the event using a cumsum filter.',
        type=float,
        default=2.0)

    parser.add_argument(
        '--hold_period',
        help='The number of days that an assert is to be held, defines the time index of the vertical barrier',
        type=int,
        default=5
    )

    parser.add_argument(
        '--target',
        help='The multiple of std of daily return that used to determine the target price, profit take/stop loss.',
        type=float,
        default=2.0
    )

    parser.add_argument(
        '--year_decay',
        help='The percentage decay of the weights for the data samples that happens each year',
        type=float,
        default=10.0
    )

    args = parser.parse_args()

    # execute
    main(args.folder, args.e_criteria, args.hold_period, args.target, args.year_decay)