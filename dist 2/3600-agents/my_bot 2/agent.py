from collections.abc import Callable
from collections import deque
from typing import List, Tuple
import random

from game import board, move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, MoveType
from .rat_belief import RatBelief


# ---------------------------------------------------------------------------
# Tunable weights
# ---------------------------------------------------------------------------
W_CARPET_POTENTIAL  = 0.8
W_RAT_EV            = 0.4
W_PRIMED_OWNED      = 0.3
W_MOBILITY          = 0.1
W_OSCILLATION       = 8.0
W_OSCILLATION_2     = 4.0
W_OSCILLATION_3     = 3.0
W_REVISIT           = 2.0
SEARCH_COOLDOWN     = 3
HISTORY_LEN         = 6
TIME_BUFFER         = 3.0   # seconds to always keep in reserve


class PlayerAgent:
    """
    Iterative-deepening Alpha-Beta Minimax with HMM rat tracking.

    Perspective convention:
      All boards passed into _max_value / _min_value are from OUR (player A)
      perspective — player_worker == us, opponent_worker == them.
      _min_value temporarily reverses perspective to get / apply opponent moves,
      then restores before returning.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.pos_history = deque(maxlen=HISTORY_LEN)
        self.search_cooldown = 0

    def commentate(self):
        return "Iterative Deepening Alpha-Beta + HMM"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()

        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)
        self.search_cooldown = max(0, self.search_cooldown - 1)

        # ----------------------------------------------------------------
        # 1. Rat search
        # ----------------------------------------------------------------
        best_idx, _ = self.hmm.get_best_guess()
        turns_used = 40 - board.player_worker.turns_left
        if (self.hmm.search_ev(best_idx) > 0
                and turns_used >= 5
                and self.search_cooldown == 0):
            self.search_cooldown = SEARCH_COOLDOWN
            self._update_history(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        # ----------------------------------------------------------------
        # 2. Iterative Deepening Alpha-Beta
        # ----------------------------------------------------------------
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not valid_moves:
            if self.search_cooldown == 0:
                self.search_cooldown = SEARCH_COOLDOWN
            self._update_history(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        valid_moves = self._order_moves(valid_moves)

        # Time budget: spread remaining time evenly, keep buffer
        turns_left = board.player_worker.turns_left
        available = time_left() - TIME_BUFFER
        time_per_turn = available / max(turns_left, 1)
        # Use at most 40% of per-turn budget (leave room for later turns)
        turn_budget = min(time_per_turn * 0.4, 5.0)
        deadline = time_left() - turn_budget

        best_move = valid_moves[0]
        for depth in range(2, 7):
            if time_left() < deadline + 0.3:
                break
            result = self._search_root(board, valid_moves, depth, time_left, deadline)
            if result is not None:
                best_move = result
            else:
                break  # timed out mid-search

        self._update_history(my_pos)
        return best_move

    # ------------------------------------------------------------------
    # Root search — returns best move or None if timed out
    # ------------------------------------------------------------------
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

            # child is from our perspective; opponent moves next
            score = self._min_value(child, alpha, beta, depth - 1, time_left, deadline)
            if score is None:
                return None  # timed out

            if score > best_score:
                best_score = score
                best_move = m

            alpha = max(alpha, best_score)

        return best_move

    # ------------------------------------------------------------------
    # MAX node — our move (board from our perspective)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # MIN node — opponent's move.
    # board is from OUR perspective.  We reverse it, get opponent moves,
    # apply each, then reverse the child back before recursing into max.
    # ------------------------------------------------------------------
    def _min_value(self, board, alpha, beta, depth, time_left, deadline):
        if time_left() < deadline:
            return None
        if depth == 0:
            return self._evaluate(board)

        # Flip to opponent's perspective to get their legal moves
        board.reverse_perspective()
        opp_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        opp_moves = self._order_moves(opp_moves)

        if not opp_moves:
            board.reverse_perspective()  # restore
            return self._evaluate(board)

        worst = float('inf')
        for em in opp_moves:
            if time_left() < deadline:
                board.reverse_perspective()  # restore before returning
                return None

            # board is currently from opponent's perspective; forecast their move
            child = board.forecast_move(em, check_ok=False)
            if child is None:
                continue

            # child is from opponent's perspective after they moved.
            # Flip back to our perspective before evaluating / recursing.
            child.reverse_perspective()

            val = self._max_value(child, alpha, beta, depth - 1, time_left, deadline)
            if val is None:
                board.reverse_perspective()  # restore before returning
                return None

            worst = min(worst, val)
            beta = min(beta, worst)
            if alpha >= beta:
                break

        board.reverse_perspective()  # restore to our perspective
        return worst

    # ------------------------------------------------------------------
    # Move ordering: high-value carpet rolls first, then primes, then plains
    # ------------------------------------------------------------------
    def _order_moves(self, moves):
        def priority(m):
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts <= 0:
                    return (2, 0)  # demote length-1 carpets (-1 pt) to plain tier
                return (0, -pts)
            elif m.move_type == MoveType.PRIME:
                return (1, 0)
            else:
                return (2, 0)
        return sorted(moves, key=priority)

    # ------------------------------------------------------------------
    # Heuristic — board must be from OUR perspective
    # ------------------------------------------------------------------
    def _evaluate(self, board):
        my_pts  = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        score   = float(my_pts - opp_pts)

        my_pos   = board.player_worker.get_location()
        opp_pos  = board.opponent_worker.get_location()
        turns_left = board.player_worker.turns_left

        # Carpet potential — more important early game
        carpet_w = W_CARPET_POTENTIAL * (turns_left / 40.0 + 0.5)
        my_cp  = self._carpet_potential(board, my_pos)
        opp_cp = self._carpet_potential(board, opp_pos)
        score += carpet_w * (my_cp - opp_cp)

        # Rat EV
        if self.search_cooldown == 0:
            best_idx, _ = self.hmm.get_best_guess()
            ev = self.hmm.search_ev(best_idx)
            score += W_RAT_EV * max(ev, 0.0)

        # Primed squares nearby
        my_p  = self._adjacent_primed(board, my_pos)
        opp_p = self._adjacent_primed(board, opp_pos)
        score += W_PRIMED_OWNED * (my_p - opp_p)

        # Mobility
        my_mv  = len(board.get_valid_moves(enemy=False, exclude_search=True))
        opp_mv = len(board.get_valid_moves(enemy=True,  exclude_search=True))
        score += W_MOBILITY * (my_mv - opp_mv)

        # Oscillation penalties (deeper history)
        hist = list(self.pos_history)
        if len(hist) >= 2 and my_pos == hist[-2]:
            score -= W_OSCILLATION
        if len(hist) >= 3 and my_pos == hist[-3]:
            score -= W_OSCILLATION_2
        if len(hist) >= 4 and my_pos == hist[-4]:
            score -= W_OSCILLATION_3
        # General revisit penalty
        revisit_count = sum(1 for p in hist if p == my_pos)
        score -= W_REVISIT * revisit_count

        return score

    def _carpet_potential(self, board, pos):
        total = 0.0
        x, y = pos
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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

    def _update_history(self, pos):
        self.pos_history.append(pos)