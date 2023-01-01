import unittest

from ..models.board import Board
from ..models.util import action_to_move, move_to_action, INDEX_TO_DIRECTION


class TestSantoriniGameLogic(unittest.TestCase):
    def test_detect_win_basic(self):
        board = Board()
        self.assertEqual(-1, board.has_won(), "no winners on init.")
        board.board_height += 1
        self.assertEqual(-1, board.has_won(), "no winners on level one.")
        board.board_height += 1
        self.assertEqual(-1, board.has_won(), "no winners on level two.")
        board.board_height += 1
        self.assertNotEqual(-1, board.has_won(), "winners on level three.")

    def test_detect_win_on_worker_jump(self):
        board = Board()
        # if no worker is at (0, 0), move one there

        TestSantoriniGameLogic._move_worker(board, location=(0, 0))

        self.assertEqual(-1, board.has_won(), "no winners on init.")
        board.board_height[0][0] += 1
        self.assertEqual(-1, board.has_won(), "no winners on level one.")
        board.board_height[0][0] += 1
        self.assertEqual(-1, board.has_won(), "no winners on level two.")
        board.board_height[0][0] += 1
        self.assertEqual(0, board.has_won(), "winners on level three.")

    def test_detect_win_on_worker_move_illegal_build(self):
        # once a move to level 3 is accomplished, the build part of the action doesn't matter
        board = TestSantoriniGameLogic._one_move_win_board()
        # no winners yet
        self.assertEqual(-1, board.has_won())

        # every move with worker 0 is a legal winning move, irrespective of build step
        for action in range(64):
            self.assertTrue(board.is_legal_action(action, 0))
            board.move_and_build(action, 0)
            self.assertEqual(0, board.has_won())
            board = TestSantoriniGameLogic._one_move_win_board()

    def test_no_win_on_height_zero(self):
        board = TestSantoriniGameLogic._one_move_win_board()
        # no winners yet
        self.assertEqual(-1, board.has_won())

        # all other workers have no winning move
        for action in range(65, 128):
            # player 0 worker 1
            if board.is_legal_action(action, 0):
                board.move_and_build(action, 0)
            self.assertEqual(-1, board.has_won())
            board = TestSantoriniGameLogic._one_move_win_board()
        for action in range(128):
            # player 1 both workers
            if board.is_legal_action(action, 1):
                board.move_and_build(action, 1)
            self.assertEqual(-1, board.has_won())
            board = TestSantoriniGameLogic._one_move_win_board()

    def test_no_legal_actions(self):
        board = TestSantoriniGameLogic._trapped_board()

        for player_id in range(2):
            self.assertFalse(board.any_legal_moves(player_id))
            for action in range(128):
                self.assertFalse(board.is_legal_action(action, player_id))

        # check for level 2 towers
        for i in range(5):
            for j in range(5):
                if board.board_height[i][j] != 0:
                    board.board_height[i][j] = 2

        for player_id in range(2):
            self.assertFalse(board.any_legal_moves(player_id))
            for action in range(128):
                self.assertFalse(board.is_legal_action(action, player_id))

        # check for level 1 towers

        for i in range(5):
            for j in range(5):
                if board.board_height[i][j] != 0:
                    board.board_height[i][j] = 1

        for player_id in range(2):
            for action in range(128):
                self.assertTrue(board.any_legal_moves(player_id))

    def test_legal_move_count(self):
        # 3 move directions and it's a winning move so build doesn't matter: 3 * 8 * 4
        board = TestSantoriniGameLogic._trapped_board(corner_height=2)

        legal_action_count = 0
        for player_id in range(2):
            for action in range(128):
                if board.is_legal_action(action, player_id):
                    legal_action_count += 1
        # 3 move directions * all build directions * 4 workers
        self.assertEqual(legal_action_count, 3 * 8 * 4)

    def test_legal_move_and_build_count(self):
        # only three moves and either 5 or 8 build directions in a trapped board with worker tower height of 1 and rest 2
        board = TestSantoriniGameLogic._trapped_board(corner_height=1)
        for i in range(5):
            for j in range(5):
                if board.board_height[i][j] != 1:
                    board.board_height[i][j] = 2

        legal_action_count = 0
        for player_id in range(2):
            for action in range(128):
                if board.is_legal_action(action, player_id):
                    legal_action_count += 1
        # 5 build directions for 2 cardinal and 8 build directions for 1 diagonal, for 4 corner worker pieces
        self.assertEqual(legal_action_count, (5 + 8 + 5) * 4)

    def test_is_occupied(self):
        board = Board()
        TestSantoriniGameLogic._move_worker(board, 0, 0, location=(0, 0))
        TestSantoriniGameLogic._move_worker(board, 0, 1, location=(1, 0))  # move worker adjacent
        TestSantoriniGameLogic._move_worker(board, 1, 0, location=(0, 4))
        TestSantoriniGameLogic._move_worker(board, 1, 1, location=(4, 4))

        board.board_height[0][1] = 4  # dome
        from_coordinate = (0, 0)

        self.assertTrue(board._is_occupied((0, 1)))   # dome
        self.assertFalse(board._can_move(from_coordinate, (0, 1)))

        self.assertTrue(board._is_occupied((1, 0)))   # worker
        self.assertFalse(board._can_move(from_coordinate, (1, 0)))

        self.assertFalse(board._is_occupied((1, 1)))  # empty
        self.assertTrue(board._can_move(from_coordinate, (1, 1)))

    def test_legal_action_mask_one_legal_move(self):
        board = TestSantoriniGameLogic._trapped_board()

        for i in range(5):
            for j in range(5):
                if board.board_height[i][j] == 3:
                    board.board_height[i][j] = 4  # replace with dome

        board.board_height[0][1] = 0

        # no legal move for player 1
        self.assertFalse(board.any_legal_moves(1))
        self.assertEqual([], board.get_legal_moves(1))
        for action in range(128):
            self.assertFalse(board.is_legal_action(action, 1))

        # no legal move for player 0 worker 1
        for action in range(64, 128):
            self.assertFalse(board.is_legal_action(action, 0))

        # one legal move for player 0 worker 0
        self.assertTrue(board.any_legal_moves(0))
        # player 0, worker 0, move direction E, build direction W
        self.assertEqual([22], board.get_legal_moves(0))

    def test_legal_action_mask_four_legal_moves(self):
        board = TestSantoriniGameLogic._trapped_board()

        for i in range(5):
            for j in range(5):
                if board.board_height[i][j] == 3:
                    board.board_height[i][j] = 4  # replace with dome

        board.board_height[0][1] = 0
        board.board_height[1][1] = 0

        # no legal move for player 1
        self.assertFalse(board.any_legal_moves(1))
        self.assertEqual([], board.get_legal_moves(1))
        for action in range(128):
            self.assertFalse(board.is_legal_action(action, 1))

        # no legal move for player 0 worker 1
        for action in range(64, 128):
            self.assertFalse(board.is_legal_action(action, 0))

        # four legal moves for player 0 worker 0
        self.assertTrue(board.any_legal_moves(0))
        self.assertEqual(4, len(board.get_legal_moves(0)))
        # player 0, worker 0, move direction E, build direction S
        self.assertIn(20, board.get_legal_moves(0))
        # player 0, worker 0, move direction E, build direction W
        self.assertIn(22, board.get_legal_moves(0))
        # player 0, worker 0, move direction SE, build direction N
        self.assertIn(24, board.get_legal_moves(0))
        # player 0, worker 0, move direction SE, build direction NW
        self.assertIn(31, board.get_legal_moves(0))

    def test_workers_trapped(self):
        board = TestSantoriniGameLogic._trapped_board()

        # no legal moves for either player
        self.assertFalse(board.any_legal_moves(0))
        self.assertFalse(board.any_legal_moves(1))

        # since both players are trapped, choice of winner is arbitrary
        self.assertNotEqual(-1, board.has_won())

    @staticmethod
    def _one_move_win_board():
        # surround player 0 worker 0 with level 3 towers in all 8 directions
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
    def _trapped_board(corner_height=0):
        # trap all workers in all corners, blocked by towers too high to legally climb
        board = Board()
        # move 4 workers to 4 corners of the board
        TestSantoriniGameLogic._move_worker(board, 0, 0, location=(0, 0))
        TestSantoriniGameLogic._move_worker(board, 0, 1, location=(4, 0))
        TestSantoriniGameLogic._move_worker(board, 1, 0, location=(0, 4))
        TestSantoriniGameLogic._move_worker(board, 1, 1, location=(4, 4))

        for i in range(5):
            for j in range(5):
                if i in (0, 4) and j in (0, 4):
                    board.board_height[i][j] = corner_height
                else:
                    board.board_height[i][j] = 3
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
