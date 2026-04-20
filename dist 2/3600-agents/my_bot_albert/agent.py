from collections.abc import Callable
from typing import List, Tuple
from collections import deque

from game import board, move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, MoveType
from .rat_belief import RatBelief


TIME_BUFFER = 5.0
HISTORY_LEN = 6
MAX_OPP_MOVES = 3
_FOUR_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class PlayerAgent:
    """
    Simple expectiminimax + HMM.
    
    "You will be surprised at how simple the Alberts are."
    
    - Expectiminimax (chance node averages top-3 opponent moves)
    - Very simple heuristic: points + carpet potential
    - No rat searching (Albert never searches)
    - Greedy carpet when available
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.pos_history = deque(maxlen=HISTORY_LEN)

    def commentate(self):
        return "AKIR-SIMPLE-EXPECTIMAX"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()
        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)

        turns_left = board.player_worker.turns_left

        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not valid_moves:
            best_idx, _ = self.hmm.get_best_guess()
            self.pos_history.append(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        # Greedy: take best carpet if worth 2+ pts
        best_carpet = None
        best_carpet_pts = 0
        for m in valid_moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts > best_carpet_pts:
                    best_carpet_pts = pts
                    best_carpet = m
        if best_carpet_pts >= 4:  # length 3+
            self.pos_history.append(my_pos)
            return best_carpet

        # Search
        valid_moves = self._order_moves(valid_moves)

        available = time_left() - TIME_BUFFER
        if available < 1.0:
            self.pos_history.append(my_pos)
            return valid_moves[0]

        time_per_turn = available / max(turns_left, 1)
        turn_budget = min(time_per_turn * 0.85, 6.0)
        deadline = time_left() - turn_budget

        best_move = valid_moves[0]
        for depth in range(2, 30):
            if time_left() < deadline + 0.05:
                break
            result = self._search_root(board, valid_moves, depth, time_left, deadline)
            if result is not None:
                best_move = result
            else:
                break

        self.pos_history.append(my_pos)
        return best_move

    # ------------------------------------------------------------------
    # Expectiminimax
    # ------------------------------------------------------------------

    def _search_root(self, board, moves, depth, time_left, deadline):
        best_move = None
        best_score = float('-inf')
        hist = list(self.pos_history)
        for m in moves:
            if time_left() < deadline:
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            score = self._chance_node(child, depth - 1, time_left, deadline)
            if score is None:
                return None
            # Simple oscillation check
            dest = child.player_worker.get_location()
            if len(hist) >= 2 and dest == hist[-2]:
                score -= 20.0
            if score > best_score:
                best_score = score
                best_move = m
        return best_move

    def _max_node(self, board, depth, time_left, deadline):
        if time_left() < deadline:
            return None
        if depth == 0:
            return self._evaluate(board)
        moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not moves:
            return self._evaluate(board)
        moves = self._order_moves(moves)
        best = float('-inf')
        for m in moves:
            if time_left() < deadline:
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            val = self._chance_node(child, depth - 1, time_left, deadline)
            if val is None:
                return None
            if val > best:
                best = val
        return best

    def _chance_node(self, board, depth, time_left, deadline):
        """Opponent turn as chance node — average top-3 moves."""
        if time_left() < deadline:
            return None
        if depth == 0:
            return self._evaluate(board)
        board.reverse_perspective()
        opp_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not opp_moves:
            board.reverse_perspective()
            return self._evaluate(board)
        opp_moves = self._order_moves(opp_moves)
        top = opp_moves[:MAX_OPP_MOVES]
        total = 0.0
        count = 0
        for em in top:
            if time_left() < deadline:
                board.reverse_perspective()
                return None
            child = board.forecast_move(em, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val = self._max_node(child, depth - 1, time_left, deadline)
            if val is None:
                board.reverse_perspective()
                return None
            total += val
            count += 1
        board.reverse_perspective()
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def _order_moves(self, moves):
        def priority(m):
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                return (0, -pts) if pts > 0 else (4, 0)
            elif m.move_type == MoveType.PRIME:
                return (1, 0)
            else:
                return (2, 0)
        return sorted(moves, key=priority)

    # ------------------------------------------------------------------
    # Evaluation — SIMPLE. Just like Albert.
    # ------------------------------------------------------------------

    def _evaluate(self, board):
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        turns_left = board.player_worker.turns_left
        my_pos = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        # 1. Point differential
        late = turns_left <= 15
        score = (2.5 if late else 1.0) * float(my_pts - opp_pts)

        # 2. Carpet potential
        time_w = turns_left / 40.0 + 0.5
        my_cp = self._carpet_potential(board, my_pos)
        opp_cp = self._carpet_potential(board, opp_pos)
        score += 2.0 * time_w * (my_cp - opp_cp)

        # 3. Can we prime? (on SPACE = can prime next turn)
        if board.get_cell(my_pos) == Cell.SPACE:
            score += 1.0
        if board.get_cell(opp_pos) == Cell.SPACE:
            score -= 1.0

        return score

    def _carpet_potential(self, board, pos):
        total = 0.0
        x, y = pos
        for dx, dy in _FOUR_DIRS:
            run = 0
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                if board.get_cell((nx, ny)) == Cell.PRIMED:
                    run += 1
                    nx += dx
                    ny += dy
                else:
                    break
            if run >= 2:
                total += CARPET_POINTS_TABLE.get(run, 0)
        return total
