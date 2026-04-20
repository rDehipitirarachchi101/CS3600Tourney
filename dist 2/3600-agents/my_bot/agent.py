from collections.abc import Callable
from typing import List, Tuple
from collections import deque

from game import board, move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, MoveType, Direction, loc_after_direction
from .rat_belief import RatBelief


# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------
# Only search when EV > opportunity cost of priming (+1 pt guaranteed)
# search_ev = 6p - 2 > 1.0  =>  p > 0.5
SEARCH_EV_MIN       = 2.0
SEARCH_COOLDOWN     = 5
NO_SEARCH_TURNS     = 10
TIME_BUFFER         = 5.0
HISTORY_LEN         = 6

_FOUR_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class PlayerAgent:
    """
    v7.0 — Adversarial carpet strategy + smart rat guessing.
    
    Core principles:
    1. Always prime when possible (guaranteed +1)
    2. Carpet long runs immediately (greedy — don't waste turns)
    3. Block opponent by priming in their path
    4. Search for rat only when EV > opportunity cost of priming
    5. Use opponent search info to sharpen HMM belief
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.pos_history = deque(maxlen=HISTORY_LEN)
        self.search_cooldown = 0

    def commentate(self):
        return "AKIR-v9.0"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()

        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)
        self.search_cooldown = max(0, self.search_cooldown - 1)

        best_idx, best_p = self.hmm.get_best_guess()
        turns_left = board.player_worker.turns_left
        turns_used = 40 - turns_left

        # --- Smart rat search decision ---
        # Compare search EV against what we'd gain from best available move
        search_ev = self.hmm.search_ev(best_idx)
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)

        if (search_ev > SEARCH_EV_MIN
                and turns_used >= NO_SEARCH_TURNS
                and self.search_cooldown == 0):
            # Check: is there a carpet >= 3 available? If so, carpet is better
            best_carpet_pts = 0
            for m in valid_moves:
                if m.move_type == MoveType.CARPET:
                    pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                    if pts > best_carpet_pts:
                        best_carpet_pts = pts
            # Only search if it beats the best immediate carpet
            if search_ev > best_carpet_pts:
                self.search_cooldown = SEARCH_COOLDOWN
                self.pos_history.append(my_pos)
                x, y = best_idx % 8, best_idx // 8
                return move.Move.search((x, y))

        # --- No valid movement moves? Forced search ---
        if not valid_moves:
            self.search_cooldown = max(self.search_cooldown, SEARCH_COOLDOWN)
            self.pos_history.append(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        # --- Greedy: take any carpet >= 3 immediately ---
        best_carpet = None
        best_carpet_pts = 0
        for m in valid_moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts > best_carpet_pts:
                    best_carpet_pts = pts
                    best_carpet = m
        if best_carpet_pts >= 6:  # length 4+ = 6+ pts; let search handle 3-cell decisions
            self.pos_history.append(my_pos)
            return best_carpet

        # --- Alpha-beta search for the best move ---
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
    # Alpha-Beta
    # ------------------------------------------------------------------

    def _search_root(self, board, moves, depth, time_left, deadline):
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        hist = list(self.pos_history)
        for m in moves:
            if time_left() < deadline:
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            score = self._min_value(child, alpha, beta, depth - 1, time_left, deadline)
            if score is None:
                return None
            # Oscillation penalty
            dest = child.player_worker.get_location()
            if len(hist) >= 2 and dest == hist[-2]:
                score -= 25.0
            elif len(hist) >= 3 and dest == hist[-3]:
                score -= 15.0
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)
        return best_move

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
    # Evaluation — adversarial carpet + territory control
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

        # 2. Carpet potential — what can each player carpet right now?
        #    This captures both "I should carpet my runs" and
        #    "opponent might steal my primed squares"
        time_w = turns_left / 40.0 + 0.5
        my_cp = self._best_carpet_value(board, my_pos)
        opp_cp = self._best_carpet_value(board, opp_pos)
        score += 1.8 * time_w * (my_cp - opp_cp)

        # 3. Can-prime bonus: you can only prime from SPACE cells
        #    This is the #1 factor for reducing plain moves
        my_cell = board.get_cell(my_pos)
        opp_cell = board.get_cell(opp_pos)
        prime_bonus = 1.2 * time_w
        if my_cell == Cell.SPACE:
            score += prime_bonus
        if opp_cell == Cell.SPACE:
            score -= prime_bonus

        # 4. Primed run extension: reward being next to a primed line
        #    that we can extend (builds toward big carpet rolls)
        my_ext = self._extension_potential(board, my_pos)
        opp_ext = self._extension_potential(board, opp_pos)
        score += 1.5 * time_w * (my_ext - opp_ext)

        # 5. Opponent blocking: primed squares block opponent movement
        #    More primed squares near opponent = they're more constrained
        opp_blocked = self._opponent_constraint(board, opp_pos)
        my_blocked = self._opponent_constraint(board, my_pos)
        score += 0.3 * (opp_blocked - my_blocked)

        return score

    # ------------------------------------------------------------------
    # Heuristic helpers — all O(1) or O(small constant)
    # ------------------------------------------------------------------

    def _best_carpet_value(self, board, pos):
        """Sum of carpet values for all runs >= 2 adjacent to pos."""
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

    def _extension_potential(self, board, pos):
        """
        If standing on SPACE, how long could a primed line through
        this cell become? Rewards positions that extend existing lines.
        Only checks horizontal and vertical — fast.
        """
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
            if total_line >= 2:
                pts = CARPET_POINTS_TABLE.get(min(total_line, 7), 0)
                if pts > best:
                    best = float(pts)
        return best

    def _opponent_constraint(self, board, pos):
        """Count how many of the 4 cardinal neighbors are blocked for movement."""
        x, y = pos
        blocked = 0
        for dx, dy in _FOUR_DIRS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= 8 or ny < 0 or ny >= 8:
                blocked += 1
            elif board.get_cell((nx, ny)) in (Cell.BLOCKED, Cell.PRIMED):
                blocked += 1
        return blocked
