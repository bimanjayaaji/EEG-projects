import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations
import matplotlib.pyplot as plt
import pandas as pd


def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    
    params.serial_port = '/dev/ttyACM0'
    params.timeout = 15
    board_id = BoardIds.GANGLION_BOARD.value
    
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = int(board_descr['sampling_rate'])
    board = BoardShim(board_id, params)

    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    data = pd.read_csv('/home/bimanjaya/learner/TA/brainflow/EEG-projects/src/data/mindful.csv')
    data = data.transpose()
    data = data.to_numpy()

    windowed = DataFilter.get_window(WindowOperations.HAMMING.value,len(data))
    print(windowed)

    eeg_channels = board_descr['eeg_channels']

    eeg_channel = eeg_channels[1]

    DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)

    psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate,
                                   WindowOperations.HAMMING.value)
    print('')
    print(f'psd : {psd}')

    # get band power of alpha waves (7-13 Hz)
    band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
    print('')
    print(f'band_power_alpha : {band_power_alpha}')

    # get band power of beta waves (14-30 Hz)
    band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)
    print('')
    print(f'band_power_beta : {band_power_beta}')

    # ratio of alpha and beta
    print('')
    print("alpha/beta:%f", band_power_alpha / band_power_beta)

    plt.plot(psd[1],psd[0])
    plt.show()

if __name__ == "__main__":
    main()