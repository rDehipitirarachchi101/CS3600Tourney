from collections.abc import Callable
from typing import List, Tuple
import random

from game import board, move, enums
from game.enums import Cell, CARPET_POINTS_TABLE
from .rat_belief import RatBelief


# ---------------------------------------------------------------------------
# Tunable weights for the heuristic
# ---------------------------------------------------------------------------
W_CARPET_POTENTIAL  = 0.6   # weight on reachable carpet-roll value
W_RAT_EV            = 0.4   # weight on rat expected-value contribution
W_PRIMED_OWNED      = 0.2   # reward for primed squares near our position
W_OSCILLATION       = 8.0   # penalty for returning to last position (A→B→A)
W_OSCILLATION_2     = 4.0   # softer penalty for two-step repeat (A→B→C→A)
SEARCH_COOLDOWN     = 3     # turns to wait between searches


class PlayerAgent:
    """
    Depth-2 Alpha-Beta Minimax agent with HMM rat tracking.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.last_pos = None
        self.second_last_pos = None
        self.search_cooldown = 0   # turns remaining before we can search again

    def commentate(self):
        return "Alpha-Beta Minimax + HMM rat tracker"

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()

        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)

        # Tick down the search cooldown every turn
        self.search_cooldown = max(0, self.search_cooldown - 1)

        # ----------------------------------------------------------------
        # 1. Opportunistic rat search (EV > 0  ⟺  P > 1/3)
        #    - Wait 5 turns for HMM belief to stabilize
        #    - Respect cooldown so we don't hammer the same cell repeatedly
        # ----------------------------------------------------------------
        best_idx, best_p = self.hmm.get_best_guess()
        turns_used = 40 - board.player_worker.turns_left
        if (self.hmm.search_ev(best_idx) > 0
                and turns_used >= 5
                and self.search_cooldown == 0):
            x, y = best_idx % 8, best_idx // 8
            self.search_cooldown = SEARCH_COOLDOWN
            self._update_history(my_pos)
            return move.Move.search((x, y))

        # ----------------------------------------------------------------
        # 2. Alpha-Beta Minimax (depth 2)
        # ----------------------------------------------------------------
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not valid_moves:
            # Only fall back to search if cooldown allows, else stay put
            if self.search_cooldown == 0:
                self.search_cooldown = SEARCH_COOLDOWN
                self._update_history(my_pos)
                x, y = best_idx % 8, best_idx // 8
                return move.Move.search((x, y))
            # Extremely rare — no moves and in cooldown; forced search anyway
            self._update_history(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for m in valid_moves:
            if time_left() < 1.0:
                break

            sim_board = board.forecast_move(m, check_ok=False)
            if sim_board is None:
                continue

            score = self._min_value(sim_board, alpha, beta, time_left)

            if score > best_score:
                best_score = score
                best_move = m

            alpha = max(alpha, best_score)

        if best_move is None:
            best_move = random.choice(valid_moves)

        self._update_history(my_pos)
        return best_move

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def _update_history(self, pos):
        """Shift position history forward by one step."""
        self.second_last_pos = self.last_pos
        self.last_pos = pos

    # ------------------------------------------------------------------
    # Alpha-Beta helpers
    # ------------------------------------------------------------------

    def _min_value(self, sim_board, alpha, beta, time_left):
        """MIN node: opponent picks the move that minimises our score."""
        enemy_moves = sim_board.get_valid_moves(enemy=True, exclude_search=True)

        if not enemy_moves:
            return self._evaluate(sim_board)

        min_score = float('inf')
        for em in enemy_moves:
            if time_left() < 1.0:
                break

            sim_board_2 = sim_board.forecast_move(em, check_ok=False)
            if sim_board_2 is None:
                continue

            score = self._evaluate(sim_board_2)
            min_score = min(min_score, score)

            if min_score <= alpha:   # alpha cut-off
                return min_score

            beta = min(beta, min_score)

        return min_score

    # ------------------------------------------------------------------
    # Heuristic evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, sim_board):
        """
        Evaluate a leaf node from our perspective.

        Components:
          - Point differential (already-scored points)
          - Carpet potential: value of carpet rolls reachable from our position
          - Rat EV: expected value of our best available search
          - Primed squares near us vs near opponent
          - Oscillation penalties: discourage A→B→A and A→B→C→A patterns
        """
        my_pts  = sim_board.player_worker.get_points()
        opp_pts = sim_board.opponent_worker.get_points()
        score   = float(my_pts - opp_pts)

        my_pos  = sim_board.player_worker.get_location()
        opp_pos = sim_board.opponent_worker.get_location()

        # Carpet potential differential
        my_carpet  = self._carpet_potential(sim_board, my_pos)
        opp_carpet = self._carpet_potential(sim_board, opp_pos)
        score += W_CARPET_POTENTIAL * (my_carpet - opp_carpet)

        # Rat EV — only count if cooldown would allow a search soon
        best_idx, _ = self.hmm.get_best_guess()
        ev = self.hmm.search_ev(best_idx)
        if self.search_cooldown == 0:
            score += W_RAT_EV * max(ev, 0.0)

        # Primed-square proximity: reward primed squares adjacent to us
        my_primed  = self._adjacent_primed(sim_board, my_pos)
        opp_primed = self._adjacent_primed(sim_board, opp_pos)
        score += W_PRIMED_OWNED * (my_primed - opp_primed)

        # Oscillation penalties
        if self.last_pos is not None and my_pos == self.last_pos:
            score -= W_OSCILLATION          # A→B→A: strong penalty
        if self.second_last_pos is not None and my_pos == self.second_last_pos:
            score -= W_OSCILLATION_2        # A→B→C→A: softer penalty

        return score

    def _carpet_potential(self, sim_board, pos):
        """
        Sum of carpet-roll points available from pos in all four directions.
        Looks at contiguous primed squares starting one step from pos.
        """
        total = 0.0
        x, y = pos
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            run = 0
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                if sim_board.get_cell((nx, ny)) == Cell.PRIMED:
                    run += 1
                    nx += dx
                    ny += dy
                else:
                    break
            if run > 0:
                total += CARPET_POINTS_TABLE.get(run, 0)
        return total

    def _adjacent_primed(self, sim_board, pos):
        """Count primed squares within Manhattan distance 2 of pos."""
        x, y = pos
        count = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if sim_board.get_cell((nx, ny)) == Cell.PRIMED:
                        count += 1
        return count