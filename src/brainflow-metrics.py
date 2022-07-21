import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams


def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyACM0'
    params.timeout = 15

    board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    board.prepare_session()
    board.start_stream(45000) #, args.streamer_params)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(10)  # recommended window size for eeg metric calculation is at least 4 seconds, bigger is better
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
    print('')
    print(f'eeg_channels : {eeg_channels}')

    bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
    print('')
    print(f'bands : {bands}')
    print(f'bands type : {type(bands)}')
    print(f'bands len : {len(bands)}')
    print(f'bands[0] : {bands[0]}')
    print(f'bands [0] shape : {bands[0].shape}')

    feature_vector = bands[0]
    print('')
    print(f'feature_vector : {feature_vector}')

    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    print('')
    print('Mindfulness: %s' % str(mindfulness.predict(feature_vector)))
    mindfulness.release()

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
