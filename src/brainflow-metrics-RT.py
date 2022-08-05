from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
import argparse
import time

def main():
    delay = 5

    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyACM0'
    params.timeout = 15

    board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    timestamp = BoardShim.get_timestamp_channel(master_board_id)
    board_descr = BoardShim.get_board_descr(master_board_id)
    eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
    
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')

    keep_alive = True

    while keep_alive == True:
        while board.get_board_data_count() < 1000:
            print('waiting 5 seconds')
        data = board.get_current_board_data(1000)
        bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
        feature_vector = bands[0]

        # mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
        #                                       BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
        # mindfulness = MLModel(mindfulness_params)
        # mindfulness.prepare()
        # print('')
        # print('Mindfulness: %s' % str(mindfulness.predict(feature_vector)))
        # mindfulness.release()

        restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
        restfulness = MLModel(restfulness_params)
        restfulness.prepare()
        print('')
        print('Restfulness: %s' % str(restfulness.predict(feature_vector)))
        print('')
        restfulness.release()

if __name__ == "__main__":
    main()
