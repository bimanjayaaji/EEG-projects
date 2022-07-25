from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations, FilterTypes
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
import matplotlib.pyplot as plt
from joblib import dump,load
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def main():
    loaded = load('/home/bimanjaya/learner/TA/brainflow/EEG-projects/src/model1.joblib') 
    delay = 5

    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()

    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyACM0'
    params.timeout = 15

    board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    timestamp = BoardShim.get_timestamp_channel(master_board_id)
    board_descr = BoardShim.get_board_descr(master_board_id)

    eeg_channels = board_descr['eeg_channels']
    eeg_channel1 = eeg_channels[0]
    eeg_channel2 = eeg_channels[1]
    eeg_channel3 = eeg_channels[2]
    eeg_channel4 = eeg_channels[3]
    
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')

    keep_alive = True

    while keep_alive == True:
        while board.get_board_data_count() < 1000:
            print('waiting 5 seconds')
        data = board.get_current_board_data(1000)
        total_iterate = data.shape[1] // (sampling_rate*delay)

        # dict = {'alpha1':[],'alpha2':[],'alpha3':[],'alpha4':[],
        # 'beta1':[],'beta2':[],'beta3':[],'beta4':[]}
        # df = pd.DataFrame(dict)

        print('')
        print('---------- start iteration ----------')
        print('')
        for x in range(total_iterate):
            ch1_iter = data[eeg_channel1][(x*1000):((x+1)*1000)]
            ch2_iter = data[eeg_channel2][(x*1000):((x+1)*1000)]
            ch3_iter = data[eeg_channel3][(x*1000):((x+1)*1000)]
            ch4_iter = data[eeg_channel4][(x*1000):((x+1)*1000)]
            
            DataFilter.detrend(ch1_iter, DetrendOperations.LINEAR.value)
            DataFilter.detrend(ch2_iter, DetrendOperations.LINEAR.value)
            DataFilter.detrend(ch3_iter, DetrendOperations.LINEAR.value)
            DataFilter.detrend(ch4_iter, DetrendOperations.LINEAR.value)
            
            DataFilter.perform_bandpass(ch1_iter, BoardShim.get_sampling_rate(master_board_id), 5.0, 35.0, 4,
                                                FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandpass(ch2_iter, BoardShim.get_sampling_rate(master_board_id), 5.0, 35.0, 4,
                                                FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandpass(ch3_iter, BoardShim.get_sampling_rate(master_board_id), 5.0, 35.0, 4,
                                                FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandpass(ch4_iter, BoardShim.get_sampling_rate(master_board_id), 5.0, 35.0, 4,
                                                FilterTypes.BUTTERWORTH.value, 0)
            
            psd1 = DataFilter.get_psd_welch(ch1_iter, nfft, nfft // 2, sampling_rate,
                                        WindowOperations.HAMMING.value)
            psd2 = DataFilter.get_psd_welch(ch2_iter, nfft, nfft // 2, sampling_rate,
                                        WindowOperations.HAMMING.value)
            psd3 = DataFilter.get_psd_welch(ch3_iter, nfft, nfft // 2, sampling_rate,
                                        WindowOperations.HAMMING.value)
            psd4 = DataFilter.get_psd_welch(ch4_iter, nfft, nfft // 2, sampling_rate,
                                        WindowOperations.HAMMING.value)
            
            band_power_total1 = DataFilter.get_band_power(psd1, psd1[1][0], psd1[1][-1])
            band_power_total2 = DataFilter.get_band_power(psd2, psd2[1][0], psd2[1][-1])
            band_power_total3 = DataFilter.get_band_power(psd3, psd3[1][0], psd3[1][-1])
            band_power_total4 = DataFilter.get_band_power(psd4, psd4[1][0], psd4[1][-1])
            
            band_power_alpha1 = DataFilter.get_band_power(psd1, 8.0, 13.0)
            band_power_alpha2 = DataFilter.get_band_power(psd2, 8.0, 13.0)
            band_power_alpha3 = DataFilter.get_band_power(psd3, 8.0, 13.0)
            band_power_alpha4 = DataFilter.get_band_power(psd4, 8.0, 13.0)
            
            alpha_relative1 = band_power_alpha1/band_power_total1
            alpha_relative2 = band_power_alpha2/band_power_total2
            alpha_relative3 = band_power_alpha3/band_power_total3
            alpha_relative4 = band_power_alpha4/band_power_total4
            
            band_power_beta1 = DataFilter.get_band_power(psd1, 13.0, 32.0)
            band_power_beta2 = DataFilter.get_band_power(psd2, 13.0, 32.0)
            band_power_beta3 = DataFilter.get_band_power(psd3, 13.0, 32.0)
            band_power_beta4 = DataFilter.get_band_power(psd4, 13.0, 32.0)
            
            beta_relative1 = band_power_beta1/band_power_total1
            beta_relative2 = band_power_beta2/band_power_total2
            beta_relative3 = band_power_beta3/band_power_total3
            beta_relative4 = band_power_beta4/band_power_total4
            
            dataset = [alpha_relative1,alpha_relative2,
                        alpha_relative3,alpha_relative4,
                        beta_relative1,beta_relative2,
                        beta_relative3,beta_relative4]

            # dict1 = {'alpha1':[alpha_relative1],'alpha2':[alpha_relative2],
            #         'alpha3':[alpha_relative3],'alpha4':[alpha_relative4],
            #         'beta1':[beta_relative1],'beta2':[beta_relative2],
            #         'beta3':[beta_relative3],'beta4':[beta_relative4]}
            
            # df2 = pd.DataFrame(dict1)
            # df = pd.concat([df,df2],ignore_index=True)

        print(loaded.predict_proba([dataset]))


if __name__ == "__main__":
    main()

