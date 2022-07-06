from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import datetime
import time

# def date_now():
#     today = datetime.datetime.now().strftime("%Y-%m-%d")
#     today = str(today)
#     return today

def main ():
    # file = open('/home/bimanjaya/learner/TA/brainflow/EEG-projects/data_logger/'+date_now()+'2'+'.csv', 'a+', newline ='')
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyACM0'
    params.timeout = 15
    # params.mac_address = 'D6:B8:88:C4:46:70'

    BoardShim.enable_dev_board_logger()
    
    board = BoardShim(BoardIds.GANGLION_BOARD.value, params)
    board.prepare_session()

    board.start_stream () # use this for default options
    time.sleep(10)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    # print(board.get_board_data) # ADDED
    data = board.get_board_data() # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()

    # print(data)
    # print(type(data))
    # print(data.shape)


if __name__ == "__main__":
    main()