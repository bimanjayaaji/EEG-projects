import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations
import matplotlib.pyplot as plt


def main():
    BoardShim.enable_dev_board_logger()

    # use synthetic board for demo
    params = BrainFlowInputParams()
    
    # if wanna use ganglion board
    params.serial_port = '/dev/ttyACM0'
    params.timeout = 15
    board_id = BoardIds.GANGLION_BOARD.value
    
    # if wanna use syntethic board
    '''
    board_id = BoardIds.SYNTHETIC_BOARD.value
    '''
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = int(board_descr['sampling_rate'])
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(10)

    # "get_nearest_power_of_two(sampling_rate)" --> self explanatory
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    print('')
    print(f'nfft : {nfft}')
    print(f'nfft type : {type(nfft)}')

    data = board.get_board_data()
    print('')
    print(data)
    print(f'data type: {type(data)}')
    print(f'data shape: {data.shape}')
    print('')
    board.stop_stream()
    board.release_session()

    eeg_channels = board_descr['eeg_channels']
    print('')
    print(f'eeg_channels : {eeg_channels}')

    # we'll only use channel 2 from eeg, in this case we use either ganglion or syntethic
    # second eeg channel of synthetic board is a sine wave at 10Hz, should see huge alpha
    eeg_channel = eeg_channels[3]
    print('')
    print(f'eeg_channels[1] : {eeg_channels[1]}')
    print(f'data[eeg_channels[1]] : {data[eeg_channel]}')
    print(f'data[eeg_channels[1]] size : {len(data[eeg_channel])}')
    print(f'type : {type(data[eeg_channels[1]])}')

    # optional detrend
    # more references on "detrend":
    # https://scipy-lectures.org/intro/scipy/auto_examples/plot_detrend.html
    DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)

    # calculating power spectral density with Blackmand Harris method
    # more methods on brainflow :
    #   NO_WINDOW= 0
    #   HANNING= 1
    #   HAMMING= 2
    #   BLACKMAN_HARRIS= 3
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

    # fail test if ratio is not smth we expect
    # if (band_power_alpha / band_power_beta < 100):
    #     raise ValueError('Wrong Ratio')

    plt.plot(psd[1],psd[0])
    plt.show()

if __name__ == "__main__":
    main()