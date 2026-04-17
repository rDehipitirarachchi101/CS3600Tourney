from collections.abc import Callable
from typing import List, Tuple
from collections import deque

from game import board, move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, MoveType, Direction, loc_after_direction
from .rat_belief import RatBelief


# ---------------------------------------------------------------------------
# Tunable weights
# ---------------------------------------------------------------------------
W_CARPET_POTENTIAL  = 2.0
W_PRIME_EXTEND      = 1.0
W_PRIMED_OWNED      = 0.2
W_MOBILITY          = 0.05
W_RAT_EV            = 0.3
W_ON_SPACE          = 0.4

# Oscillation (root only)
W_OSCILLATION       = 25.0
W_OSCILLATION_2     = 18.0
W_OSCILLATION_3     = 10.0
W_REVISIT           = 5.0

SEARCH_COOLDOWN     = 5
SEARCH_EV_THRESHOLD = 3.0
NO_SEARCH_TURNS     = 18
TIME_BUFFER         = 5.0
HISTORY_LEN         = 6

# Expectiminimax: how many top opponent moves to consider in chance node
MAX_OPP_MOVES       = 5

_FOUR_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class PlayerAgent:
    """
    Iterative-deepening Expectiminimax with HMM rat tracking.

    Instead of assuming the opponent plays the worst-case move (minimax),
    we model the opponent's move as a chance node: we take the top-K
    opponent moves (by a quick heuristic score), assign them weights
    proportional to their quality, and compute the weighted average.

    This better handles the stochastic nature of the game — the opponent
    won't always find the optimal counter-move, and board randomness
    means worst-case analysis is overly pessimistic.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.pos_history = deque(maxlen=HISTORY_LEN)
        self.search_cooldown = 0

    def commentate(self):
        return "Expectiminimax + HMM v1"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()

        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)
        self.search_cooldown = max(0, self.search_cooldown - 1)

        best_idx, best_p = self.hmm.get_best_guess()
        turns_left = board.player_worker.turns_left
        turns_used = 40 - turns_left

        # --- Rat search: very selective ---
        search_ev = self.hmm.search_ev(best_idx)
        if (search_ev > SEARCH_EV_THRESHOLD
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

        # --- Time management ---
        available = time_left() - TIME_BUFFER
        if available < 1.0:
            self.pos_history.append(my_pos)
            return valid_moves[0]

        time_per_turn = available / max(turns_left, 1)
        turn_budget = min(time_per_turn * 0.9, 6.0)
        deadline = time_left() - turn_budget

        best_move = valid_moves[0]
        for depth in range(2, 30):
            if time_left() < deadline + 0.1:
                break
            result = self._search_root(board, valid_moves, depth, time_left, deadline)
            if result is not None:
                best_move = result
            else:
                break

        self.pos_history.append(my_pos)
        return best_move

    # ------------------------------------------------------------------
    # Expectiminimax search
    # ------------------------------------------------------------------

    def _search_root(self, board, moves, depth, time_left, deadline):
        best_move = None
        best_score = float('-inf')

        for m in moves:
            if time_left() < deadline:
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            score = self._chance_node(child, depth - 1, time_left, deadline)
            if score is None:
                return None

            # Oscillation penalty at root
            dest = child.player_worker.get_location()
            score += self._oscillation_penalty(dest)

            if score > best_score:
                best_score = score
                best_move = m

        return best_move

    def _max_node(self, board, depth, time_left, deadline):
        """Our turn: pick the move that maximizes score."""
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
        """
        Opponent's turn modeled as a chance node.

        Instead of pure min (assuming optimal opponent), we:
        1. Get opponent's valid moves
        2. Quick-score each move for the opponent
        3. Take the top-K moves
        4. Weight them by softmax of their quick scores
        5. Return the weighted average of the resulting states

        This is the key difference from minimax — it models the opponent
        as "probably good but not perfect", which better matches reality.
        """
        if time_left() < deadline:
            return None
        if depth == 0:
            return self._evaluate(board)

        board.reverse_perspective()
        opp_moves = board.get_valid_moves(enemy=False, exclude_search=True)

        if not opp_moves:
            board.reverse_perspective()
            return self._evaluate(board)

        # Quick-score each opponent move (from opponent's perspective)
        scored = []
        for m in opp_moves:
            scored.append((self._quick_score_move(board, m), m))

        # Sort by score descending (best for opponent first)
        scored.sort(key=lambda x: -x[0])

        # Take top-K
        top_moves = scored[:MAX_OPP_MOVES]

        # Compute weights via softmax-like weighting
        # Higher-scored moves get more weight
        scores_list = [s for s, _ in top_moves]
        max_s = scores_list[0]
        weights = []
        for s in scores_list:
            # Simple exponential weighting, clamped to avoid overflow
            w = 2.718 ** min(s - max_s + 2.0, 5.0)  # shift so best = e^2
            weights.append(w)
        total_w = sum(weights)
        if total_w < 1e-9:
            weights = [1.0] * len(top_moves)
            total_w = len(top_moves)

        # Weighted average of child values
        weighted_sum = 0.0
        for i, (_, m) in enumerate(top_moves):
            if time_left() < deadline:
                board.reverse_perspective()
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val = self._max_node(child, depth - 1, time_left, deadline)
            if val is None:
                board.reverse_perspective()
                return None
            weighted_sum += (weights[i] / total_w) * val

        board.reverse_perspective()
        return weighted_sum

    def _quick_score_move(self, board, m):
        """
        Fast heuristic score for an opponent move (from their perspective).
        Used to weight the chance node — no deep search, just immediate value.
        """
        if m.move_type == MoveType.CARPET:
            pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
            return pts + 5.0  # carpet moves are generally strong
        elif m.move_type == MoveType.PRIME:
            return 2.0  # priming is decent
        else:
            return 0.0  # plain moves are weak

    # ------------------------------------------------------------------
    # Oscillation penalty
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def _order_moves(self, moves):
        def priority(m):
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts <= 0:
                    return (3, 0)
                return (0, -pts)
            elif m.move_type == MoveType.PRIME:
                return (1, 0)
            else:
                return (2, 0)
        return sorted(moves, key=priority)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, board):
        my_pts   = board.player_worker.get_points()
        opp_pts  = board.opponent_worker.get_points()
        turns_left = board.player_worker.turns_left

        late_game = turns_left <= 12
        score = (3.0 if late_game else 1.0) * float(my_pts - opp_pts)

        my_pos  = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        # Carpet potential
        carpet_w = W_CARPET_POTENTIAL * (turns_left / 40.0 + 0.5)
        my_cp  = self._carpet_potential(board, my_pos)
        opp_cp = self._carpet_potential(board, opp_pos)
        score += carpet_w * (my_cp - opp_cp)

        # Prime extension value
        ext_w = W_PRIME_EXTEND * (turns_left / 40.0 + 0.3)
        my_ext  = self._prime_extend_value(board, my_pos)
        opp_ext = self._prime_extend_value(board, opp_pos)
        score += ext_w * (my_ext - opp_ext)

        # Space bonus
        if board.get_cell(my_pos) == Cell.SPACE:
            score += W_ON_SPACE
        if board.get_cell(opp_pos) == Cell.SPACE:
            score -= W_ON_SPACE

        # Adjacent primed
        my_p  = self._adjacent_primed(board, my_pos)
        opp_p = self._adjacent_primed(board, opp_pos)
        score += W_PRIMED_OWNED * (my_p - opp_p)

        # Mobility
        my_mv  = len(board.get_valid_moves(enemy=False, exclude_search=True))
        opp_mv = len(board.get_valid_moves(enemy=True,  exclude_search=True))
        score += W_MOBILITY * (my_mv - opp_mv)

        # Rat EV
        if self.search_cooldown == 0 and turns_left <= 22:
            best_idx, _ = self.hmm.get_best_guess()
            score += W_RAT_EV * max(self.hmm.search_ev(best_idx), 0.0)

        return score

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

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

    def _prime_extend_value(self, board, pos):
        x, y = pos
        if board.get_cell((x, y)) != Cell.SPACE:
            return 0.0
        best = 0.0
        for dx, dy in [(1, 0), (0, 1)]:
            behind = 0
            bx, by = x - dx, y - dy
            while 0 <= bx < 8 and 0 <= by < 8 and board.get_cell((bx, by)) == Cell.PRIMED:
                behind += 1
                bx -= dx
                by -= dy
            ahead = 0
            fx, fy = x + dx, y + dy
            while 0 <= fx < 8 and 0 <= fy < 8 and board.get_cell((fx, fy)) == Cell.PRIMED:
                ahead += 1
                fx += dx
                fy += dy
            total_line = behind + 1 + ahead
            if total_line >= 3:
                pts = CARPET_POINTS_TABLE.get(min(total_line, 7), 0)
                if pts > best:
                    best = pts
        return best

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
