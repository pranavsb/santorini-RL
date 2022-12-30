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

    def test_detect_win_on_worker_move(self):
        board = Board()
        # if no worker is at (0, 0), move one there
        worker_at_origin = False
        for player_id in range(2):
            for worker in board.workers[player_id]:
                if worker.location == (0, 0):
                    worker_at_origin = True
        if not worker_at_origin:
            board.workers[0][0].location = (0, 0)

        self.assertFalse(board.has_won(), "no winners on init.")
        board.board_height[0][0] += 1
        self.assertFalse(board.has_won(), "no winners on level one.")
        board.board_height[0][0] += 1
        self.assertFalse(board.has_won(), "no winners on level two.")
        board.board_height[0][0] += 1
        self.assertTrue(board.has_won(), "winners on level three.")


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
