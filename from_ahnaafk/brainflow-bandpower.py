import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations


def main():
    BoardShim.enable_dev_board_logger()

    # use synthetic board for demo
    params = BrainFlowInputParams()
    
    # if want to use ganglion board
    '''
    params.serial_port = '/dev/ttyACM0'
    params.timeout = 15
    # change board_id to BoardIds.GANGLION_BOARD.value
    '''
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = int(board_descr['sampling_rate'])
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(10)

    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    print('')
    print(f'nfft : {nfft}')
    print(f'nfft type : {type(nfft)}')

    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = board_descr['eeg_channels']
    print('')
    print(f'eeg_channels : {eeg_channels}')

    # second eeg channel of synthetic board is a sine wave at 10Hz, should see huge alpha
    eeg_channel = eeg_channels[1]
    print('')
    print(f'eeg_channels[1] : {eeg_channels[1]}')

    # optional detrend
    DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
    psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate,
                                   WindowOperations.BLACKMAN_HARRIS.value)
    print('')
    print(f'psd : {psd}')

    band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
    print('')
    print(f'band_power_alpha : {band_power_alpha}')

    band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)
    print('')
    print(f'band_power_beta : {band_power_beta}')

    print('')
    print("alpha/beta:%f", band_power_alpha / band_power_beta)

    # fail test if ratio is not smth we expect
    if (band_power_alpha / band_power_beta < 100):
        raise ValueError('Wrong Ratio')


if __name__ == "__main__":
    main()