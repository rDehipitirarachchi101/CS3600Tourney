from collections.abc import Callable
from typing import List, Tuple
from collections import deque

from game import board, move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, MoveType
from .rat_belief import RatBelief


# ---------------------------------------------------------------------------
# Tunable weights (v4 base + targeted fixes)
# ---------------------------------------------------------------------------
W_CARPET_POTENTIAL  = 1.6
W_RAT_EV            = 0.4
W_PRIMED_OWNED      = 0.3
W_MOBILITY          = 0.1
W_ON_PRIMEABLE      = 1.5    # bumped hard: was 0.5, need to stop plain moves

# Oscillation (applied at root only)
W_OSCILLATION       = 25.0
W_OSCILLATION_2     = 18.0
W_OSCILLATION_3     = 10.0
W_REVISIT           = 5.0

SEARCH_COOLDOWN     = 5      # bumped: was 4, space out searches more
SEARCH_EV_THRESHOLD = 2.8    # bumped: was 2.5, cut ~10 searches/game to ~6-7
NO_SEARCH_TURNS     = 10
TIME_BUFFER         = 5.0
HISTORY_LEN         = 6

_FOUR_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class PlayerAgent:
    """
    Iterative-deepening Alpha-Beta with HMM rat tracking. v4.2
    - Search threshold p>0.67, allowed after turn 10
    - Priming strongly preferred over plain moves
    - Carpet potential skips length-1 runs
    - Bonus for standing on primeable space
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.pos_history = deque(maxlen=HISTORY_LEN)
        self.search_cooldown = 0

    def commentate(self):
        return "ID Alpha-Beta + HMM v4.4"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()

        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)
        self.search_cooldown = max(0, self.search_cooldown - 1)

        best_idx, _ = self.hmm.get_best_guess()
        turns_left = board.player_worker.turns_left
        turns_used = 40 - turns_left

        # Rat search — very selective, never early
        if (self.hmm.search_ev(best_idx) > SEARCH_EV_THRESHOLD
                and turns_used >= NO_SEARCH_TURNS
                and self.search_cooldown == 0):
            self.search_cooldown = SEARCH_COOLDOWN
            self.pos_history.append(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not valid_moves:
            if self.search_cooldown == 0:
                self.search_cooldown = SEARCH_COOLDOWN
            self.pos_history.append(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        valid_moves = self._order_moves(valid_moves)

        # Time management — safe
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

    def _search_root(self, board, moves, depth, time_left, deadline):
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for m in moves:
            if time_left() < deadline:
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            score = self._min_value(child, alpha, beta, depth - 1, time_left, deadline)
            if score is None:
                return None

            # Oscillation penalty at root only
            dest = child.player_worker.get_location()
            score += self._oscillation_penalty(dest)

            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)

        return best_move

    def _oscillation_penalty(self, dest):
        penalty = 0.0
        hist = list(self.pos_history)
        if len(hist) >= 2 and dest == hist[-2]:
            penalty -= W_OSCILLATION
        if len(hist) >= 3 and dest == hist[-3]:
            penalty -= W_OSCILLATION_2
        if len(hist) >= 4 and dest == hist[-4]:
            penalty -= W_OSCILLATION_3
        revisit_count = sum(1 for p in hist if p == dest)
        if revisit_count > 0:
            penalty -= W_REVISIT * revisit_count
        return penalty

    def _max_value(self, board, alpha, beta, depth, time_left, deadline):
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
            val = self._min_value(child, alpha, beta, depth - 1, time_left, deadline)
            if val is None:
                return None
            best = max(best, val)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def _min_value(self, board, alpha, beta, depth, time_left, deadline):
        if time_left() < deadline:
            return None
        if depth == 0:
            return self._evaluate(board)

        board.reverse_perspective()
        opp_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        opp_moves = self._order_moves(opp_moves)

        if not opp_moves:
            board.reverse_perspective()
            return self._evaluate(board)

        worst = float('inf')
        for em in opp_moves:
            if time_left() < deadline:
                board.reverse_perspective()
                return None
            child = board.forecast_move(em, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val = self._max_value(child, alpha, beta, depth - 1, time_left, deadline)
            if val is None:
                board.reverse_perspective()
                return None
            worst = min(worst, val)
            beta = min(beta, worst)
            if alpha >= beta:
                break

        board.reverse_perspective()
        return worst

    def _order_moves(self, moves):
        def priority(m):
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts <= 0:
                    return (3, 0)  # fix: length-1 carpet (-1 pts) worse than plain
                return (0, -pts)
            elif m.move_type == MoveType.PRIME:
                return (1, 0)
            else:
                return (2, 0)
        return sorted(moves, key=priority)

    def _evaluate(self, board):
        my_pts   = board.player_worker.get_points()
        opp_pts  = board.opponent_worker.get_points()
        turns_left = board.player_worker.turns_left

        late_game = turns_left <= 20
        score = (2.0 if late_game else 1.0) * float(my_pts - opp_pts)

        my_pos  = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        # Carpet potential — always active, scaled by time remaining
        carpet_w = W_CARPET_POTENTIAL * (turns_left / 40.0 + 0.5)
        my_cp  = self._carpet_potential(board, my_pos)
        opp_cp = self._carpet_potential(board, opp_pos)
        score += carpet_w * (my_cp - opp_cp)

        # Bonus for standing on a primeable space (encourages priming over plain)
        my_cell = board.get_cell(my_pos)
        opp_cell = board.get_cell(opp_pos)
        if my_cell == Cell.SPACE:
            score += W_ON_PRIMEABLE
        if opp_cell == Cell.SPACE:
            score -= W_ON_PRIMEABLE

        # Rat EV
        if self.search_cooldown == 0 and turns_left <= 30:
            best_idx, _ = self.hmm.get_best_guess()
            score += W_RAT_EV * max(self.hmm.search_ev(best_idx), 0.0)

        # Adjacent primed
        my_p  = self._adjacent_primed(board, my_pos)
        opp_p = self._adjacent_primed(board, opp_pos)
        score += W_PRIMED_OWNED * (my_p - opp_p)

        # Mobility
        my_mv  = len(board.get_valid_moves(enemy=False, exclude_search=True))
        opp_mv = len(board.get_valid_moves(enemy=True,  exclude_search=True))
        score += W_MOBILITY * (my_mv - opp_mv)

        return score

    def _carpet_potential(self, board, pos):
        """Score carpet runs adjacent to pos. Skip length-1 (they're -1 pts)."""
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
            if run >= 2:  # skip length-1 (-1 pts)
                total += CARPET_POINTS_TABLE.get(run, 0)
        return total

    def _adjacent_primed(self, board, pos):
        x, y = pos
        count = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if board.get_cell((nx, ny)) == Cell.PRIMED:
                        count += 1
        return count
