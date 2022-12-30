import unittest

from board import Board
from util import action_to_move, move_to_action, INDEX_TO_DIRECTION


class TestSantoriniGameLogic(unittest.TestCase):
    def test_detect_win_basic(self):
        board = Board()
        self.assertFalse(board.has_won(), "no winners on init.")
        board.board_height += 1
        self.assertFalse(board.has_won(), "no winners on level one.")
        board.board_height += 1
        self.assertFalse(board.has_won(), "no winners on level two.")
        board.board_height += 1
        self.assertTrue(board.has_won(), "winners on level three.")

    def test_detect_win_on_worker_jump(self):
        board = Board()
        # if no worker is at (0, 0), move one there

        TestSantoriniGameLogic._move_worker(board, location=(0, 0))

        self.assertFalse(board.has_won(), "no winners on init.")
        board.board_height[0][0] += 1
        self.assertFalse(board.has_won(), "no winners on level one.")
        board.board_height[0][0] += 1
        self.assertFalse(board.has_won(), "no winners on level two.")
        board.board_height[0][0] += 1
        self.assertTrue(board.has_won(), "winners on level three.")

    def test_detect_win_on_worker_move_illegal_build(self):
        board = TestSantoriniGameLogic._one_move_win_board()
        # no winners yet
        self.assertFalse(board.has_won())

        # every move with worker 0 is a legal winning move, irrespective of build step
        for action in range(64):
            self.assertTrue(board.is_legal_action(action, 0))
            board.move_and_build(action, 0)
            self.assertTrue(board.has_won())
            board = TestSantoriniGameLogic._one_move_win_board()

    def test_no_win_on_height_zero(self):
        board = TestSantoriniGameLogic._one_move_win_board()
        # no winners yet
        self.assertFalse(board.has_won())

        # all other workers have no winning move
        for action in range(65, 128):
            # player 0 worker 1
            if board.is_legal_action(action, 0):
                board.move_and_build(action, 0)
            self.assertFalse(board.has_won())
            board = TestSantoriniGameLogic._one_move_win_board()
        for action in range(128):
            # player 1 both workers
            if board.is_legal_action(action, 1):
                board.move_and_build(action, 1)
            self.assertFalse(board.has_won())
            board = TestSantoriniGameLogic._one_move_win_board()

    @staticmethod
    def _one_move_win_board():
        # once a move to level 3 is accomplished, the build part of the action doesn't matter
        board = Board()
        # move 4 workers to 4 corners of the board
        TestSantoriniGameLogic._move_worker(board, 0, 0, location=(0, 0))
        TestSantoriniGameLogic._move_worker(board, 0, 1, location=(4, 0))
        TestSantoriniGameLogic._move_worker(board, 1, 0, location=(0, 4))
        TestSantoriniGameLogic._move_worker(board, 1, 1, location=(4, 4))

        # move player 0 worker 0 to (1, 1) and surround with level 3 towers
        TestSantoriniGameLogic._move_worker(board, 0, 0, location=(1, 1))
        for i in range(3):
            for j in range(3):
                board.board_height[i][j] = 3
        board.board_height[1][1] = 2  # worker can jump one level up
        return board

    @staticmethod
    def _move_worker(board, player_id=0, worker_id=0, location=(0, 0)):
        board.workers[player_id][worker_id].location = location


class Util(unittest.TestCase):
    def test_inverse_function_action_move_transform(self):
        direction_to_index = {v: k for k, v in INDEX_TO_DIRECTION.items()}
        for action_id in range(128):
            worker_id, move_direction, build_direction = action_to_move(action_id)
            move_tuple = (worker_id, direction_to_index[move_direction], direction_to_index[build_direction])
            self.assertEqual(action_id, move_to_action(move_tuple))

    def test_move_to_action(self):
        self.assertEqual(0, move_to_action((0, 0, 0)))
        self.assertEqual(7, move_to_action((0, 0, 7)))
        self.assertEqual(16, move_to_action((0, 2, 0)))
        self.assertEqual(64, move_to_action((1, 0, 0)))
        self.assertEqual(74, move_to_action((1, 1, 2)))
        self.assertEqual(127, move_to_action((1, 7, 7)))

    def test_action_to_move(self):
        self.assertEqual((0, "N", "N"), action_to_move(0))
        self.assertEqual((0, "N", "NW"), action_to_move(7))
        self.assertEqual((0, "NE", "N"), action_to_move(8))
        self.assertEqual((0, "NW", "NW"), action_to_move(63))
        self.assertEqual((1, "N", "NW"), action_to_move(71))
        self.assertEqual((1, "NE", "N"), action_to_move(72))


if __name__ == "__main__":
    unittest.main()
