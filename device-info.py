# printing information on related board

from pprint import pprint

import brainflow
from brainflow.board_shim import BoardShim, BoardIds

board_id = BoardIds.SYNTHETIC_BOARD.value
pprint(BoardShim.get_board_descr(board_id))
