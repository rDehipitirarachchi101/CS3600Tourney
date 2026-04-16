from collections.abc import Callable
from typing import List, Tuple
from collections import deque

from game import board, move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, MoveType
from .rat_belief import RatBelief


# ---------------------------------------------------------------------------
# Tunable weights
# ---------------------------------------------------------------------------
W_CARPET_POTENTIAL  = 0.9
W_RAT_EV            = 0.4
W_PRIMED_OWNED      = 0.3
W_MOBILITY          = 0.1     # only applied near the root, see _evaluate
W_OSCILLATION       = 15.0    # applied at root only now
W_OSCILLATION_2     = 10.0
W_OSCILLATION_3     = 6.0
W_REVISIT           = 3.0
SEARCH_COOLDOWN     = 3
HISTORY_LEN         = 6

# --- Time management ---
TIME_BUFFER         = 3.0     # leave this much on the clock, always
HARD_PER_MOVE_CAP   = 6.0     # never spend more than this on a single move
SOFT_SLACK          = 0.05    # small cushion for the soft deadline check


class PlayerAgent:
    """
    Iterative-deepening Alpha-Beta with HMM rat tracking.

    v3 changes:
      1. Tapered time budget + hard per-move cap + mid-search abort.
         Previous-depth best move is kept if current depth aborts.
      2. Mobility term is only computed at the root (cheap) — removed from
         the hot evaluation path.
      3. Oscillation / revisit penalties moved to root-only, applied via
         move ordering instead of leaf eval.
      4. Late-game transition is smoothed: carpet potential tapers, no
         sharp 2x point multiplier.
      5. Length-1 carpet bug fixed: -1 point carpets are no longer
         preferred over primes in move ordering, and they don't pollute
         the carpet-potential heuristic.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.pos_history = deque(maxlen=HISTORY_LEN)
        self.search_cooldown = 0

    def commentate(self):
        return "ID Alpha-Beta + HMM v3"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()

        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)
        self.search_cooldown = max(0, self.search_cooldown - 1)

        best_idx, _ = self.hmm.get_best_guess()
        turns_used = 40 - board.player_worker.turns_left

        # Rat search
        if (self.hmm.search_ev(best_idx) > 0
                and turns_used >= 5
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

        # Root-level move preparation (ordering + oscillation filter)
        root_moves = self._prepare_root_moves(valid_moves, my_pos)

        # --- Tapered time budget ---
        turns_left = max(board.player_worker.turns_left, 1)
        available = max(time_left() - TIME_BUFFER, 0.1)
        turn_budget = min(available / (turns_left + 2), HARD_PER_MOVE_CAP)
        deadline = time_left() - turn_budget

        # Iterative deepening — keep best move from last fully-completed depth.
        best_move = root_moves[0][0]
        for depth in range(2, 30):
            if time_left() < deadline + SOFT_SLACK:
                break
            result = self._search_root(board, root_moves, depth, time_left, deadline)
            if result is not None:
                best_move = result
            else:
                break

        self.pos_history.append(my_pos)
        return best_move

    # ------------------------------------------------------------------
    # Root move preparation
    # ------------------------------------------------------------------
    def _prepare_root_moves(self, moves, my_pos):
        """
        Return list of (move, priority_tuple) sorted best-first.
        Oscillation penalty is applied here (once, against real history)
        rather than at every search leaf.
        """
        hist = list(self.pos_history)
        hist_set = set(hist)

        def dest_of(m):
            dest = getattr(m, 'destination', None)
            if dest is not None:
                return tuple(dest)
            return my_pos

        def priority(m):
            dest = dest_of(m)

            # Oscillation penalty applied once, at the root
            osc_pen = 0.0
            if len(hist) >= 2 and dest == hist[-2]:
                osc_pen += W_OSCILLATION
            if len(hist) >= 3 and dest == hist[-3]:
                osc_pen += W_OSCILLATION_2
            if len(hist) >= 4 and dest == hist[-4]:
                osc_pen += W_OSCILLATION_3
            if dest in hist_set:
                osc_pen += W_REVISIT * sum(1 for p in hist if p == dest)

            # Move-type priority with length-1 carpet fix:
            # A length-1 carpet is -1 points, so it must not sort ahead
            # of a prime (+1 point). Demote non-positive carpets to the
            # plain-step bucket.
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts <= 0:
                    type_key = (2, 0)
                else:
                    type_key = (0, -pts)
            elif m.move_type == MoveType.PRIME:
                type_key = (1, 0)
            else:
                type_key = (2, 0)

            return (osc_pen, type_key)

        scored = [(m, priority(m)) for m in moves]
        scored.sort(key=lambda x: x[1])
        return scored

    def _order_moves(self, moves):
        """Light ordering used inside the search (no oscillation logic)."""
        def priority(m):
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts <= 0:
                    return (2, 0)
                return (0, -pts)
            elif m.move_type == MoveType.PRIME:
                return (1, 0)
            else:
                return (2, 0)
        return sorted(moves, key=priority)

    # ------------------------------------------------------------------
    # Alpha-Beta search
    # ------------------------------------------------------------------
    def _search_root(self, board, root_moves, depth, time_left, deadline):
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for m, _prio in root_moves:
            if time_left() < deadline:
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            score = self._min_value(child, alpha, beta, depth - 1,
                                    time_left, deadline, is_root_child=True)
            if score is None:
                return None
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)

        return best_move

    def _max_value(self, board, alpha, beta, depth, time_left, deadline,
                   is_root_child=False):
        if time_left() < deadline:
            return None
        if depth == 0:
            return self._evaluate(board, include_mobility=is_root_child)
        moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not moves:
            return self._evaluate(board, include_mobility=is_root_child)
        moves = self._order_moves(moves)
        best = float('-inf')
        for m in moves:
            if time_left() < deadline:
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            val = self._min_value(child, alpha, beta, depth - 1,
                                  time_left, deadline, is_root_child=False)
            if val is None:
                return None
            best = max(best, val)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def _min_value(self, board, alpha, beta, depth, time_left, deadline,
                   is_root_child=False):
        if time_left() < deadline:
            return None
        if depth == 0:
            return self._evaluate(board, include_mobility=is_root_child)

        board.reverse_perspective()
        opp_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        opp_moves = self._order_moves(opp_moves)

        if not opp_moves:
            board.reverse_perspective()
            return self._evaluate(board, include_mobility=is_root_child)

        worst = float('inf')
        for em in opp_moves:
            if time_left() < deadline:
                board.reverse_perspective()
                return None
            child = board.forecast_move(em, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val = self._max_value(child, alpha, beta, depth - 1,
                                  time_left, deadline, is_root_child=False)
            if val is None:
                board.reverse_perspective()
                return None
            worst = min(worst, val)
            beta = min(beta, worst)
            if alpha >= beta:
                break

        board.reverse_perspective()
        return worst

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, board, include_mobility=False):
        my_pts   = board.player_worker.get_points()
        opp_pts  = board.opponent_worker.get_points()
        turns_left = board.player_worker.turns_left

        score = float(my_pts - opp_pts)

        my_pos  = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        # Smoothly-tapered carpet potential (never zeroed)
        carpet_w = W_CARPET_POTENTIAL * max(turns_left / 40.0, 0.2)
        my_cp  = self._carpet_potential(board, my_pos)
        opp_cp = self._carpet_potential(board, opp_pos)
        score += carpet_w * (my_cp - opp_cp)

        if self.search_cooldown == 0:
            best_idx, _ = self.hmm.get_best_guess()
            score += W_RAT_EV * max(self.hmm.search_ev(best_idx), 0.0)

        my_p  = self._adjacent_primed(board, my_pos)
        opp_p = self._adjacent_primed(board, opp_pos)
        score += W_PRIMED_OWNED * (my_p - opp_p)

        # Mobility is expensive — only at nodes directly under the root
        if include_mobility:
            my_mv  = len(board.get_valid_moves(enemy=False, exclude_search=True))
            opp_mv = len(board.get_valid_moves(enemy=True,  exclude_search=True))
            score += W_MOBILITY * (my_mv - opp_mv)

        return score

    def _carpet_potential(self, board, pos):
        """
        Sum of carpet points we could score by rolling outward in each
        cardinal direction. Only rewards runs of length >= 2 — a length-1
        run is worth -1 point and is not 'potential' worth chasing.
        """
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