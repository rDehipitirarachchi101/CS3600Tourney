from collections.abc import Callable
from typing import Tuple
from collections import deque

from game import move
from game.enums import Cell, CARPET_POINTS_TABLE, MoveType, Direction
from .rat_belief import RatBelief

# ── Search parameters ──────────────────────────────────────────────────────────
SEARCH_EV_MIN      = 3.0   # EV threshold to trigger rat search (p > 0.833 confidence)
SEARCH_COOLDOWN    = 5     # Min turns between rat searches
NO_SEARCH_TURNS    = 12    # Don't search in the opening N turns
POST_CATCH_COOLDOWN = 12   # Don't search for N turns after ANY rat catch (HMM unsettled)
TIME_BUFFER        = 4.5   # Seconds always kept in reserve
HISTORY_LEN        = 8     # Position history depth for oscillation penalty
MAX_TT_SIZE        = 80_000  # Hard cap on transposition table entries

# Carpet patience — don't greedily take a run if extending it is clearly better
PATIENCE_MIN_TURNS      = 14  # Below this many turns left, always take immediately
PATIENCE_GAIN_THRESHOLD = 4   # If extension would add <= N pts, not worth the wait

# Transposition table entry flags
_EXACT = 0   # Score is exact
_LOWER = 1   # Score is a lower bound (failed high / beta cutoff)
_UPPER = 2   # Score is an upper bound (failed low / alpha cutoff)

_DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

_DIR_DELTA = {
    Direction.UP:    (0, -1),
    Direction.RIGHT: (1,  0),
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
}


class PlayerAgent:
    """
    v8.0 — Maximum point efficiency in adversarial search.

    Core upgrades over v7.0:
    1. Negamax formulation — evaluation always from current mover's perspective,
       enabling a clean transposition table with EXACT/LOWER/UPPER bounds.
    2. Transposition table persists across iterative-deepening iterations
       within a turn; entries from shallower depths seed deeper iterations.
    3. Killer moves (2 per ply) + history heuristic — quiet moves that caused
       prior cutoffs are tried before other primes/plains.
    4. Evaluation improvements:
       - Carpet portfolio sums ALL directional runs (not just the best one)
       - 1-step reachable carpet: checks immediate neighbours for accessible runs
       - 2-step BFS mobility: measures room to manoeuvre
    5. Rat search threshold compared against the actual best reachable move
       value (carpet pts if available, else +1 for priming) rather than a
       fixed constant.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.pos_history: deque = deque(maxlen=HISTORY_LEN)
        self.search_cooldown = 0
        self._post_catch_cooldown = 0  # extra cooldown after any rat catch event
        self._tt: dict = {}       # board_key -> (depth, flag, value)
        self._killers: dict = {}  # ply -> [move, ...]  (max 2)
        self._history: dict = {}  # move_key -> int  (persists across turns)

    def commentate(self):
        return "AKIR-v11.0"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()

        # Detect rat catch events BEFORE HMM update (uses same board fields).
        # Either side catching the rat resets the rat position — HMM becomes
        # unreliable for ~POST_CATCH_COOLDOWN turns after.
        opp_s = board.opponent_search
        pl_s  = board.player_search
        if ((opp_s is not None and opp_s[0] is not None and opp_s[1]) or
                (pl_s is not None and pl_s[0] is not None and pl_s[1])):
            self._post_catch_cooldown = POST_CATCH_COOLDOWN
        else:
            self._post_catch_cooldown = max(0, self._post_catch_cooldown - 1)

        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)
        self.search_cooldown = max(0, self.search_cooldown - 1)

        best_idx, _ = self.hmm.get_best_guess()
        turns_left = board.player_worker.turns_left
        turns_used = 40 - turns_left

        # TT and killers are stale once the board position changes
        self._tt.clear()
        self._killers.clear()

        all_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        # Length-1 carpets score -1 pt — avoid when better options exist
        valid_moves = [m for m in all_moves
                       if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        # CRITICAL: never let the filter leave us with zero moves.
        # If all adjacent cells are single-primed, filtering every move as
        # a length-1 carpet triggers forced-search every turn indefinitely
        # (-2 pts per failed search vs -1 pt for a length-1 carpet that
        # at least moves us to a new position).
        if not valid_moves:
            valid_moves = all_moves

        # ── Rat search ─────────────────────────────────────────────────────────
        search_ev = self.hmm.search_ev(best_idx)
        if (search_ev > SEARCH_EV_MIN
                and turns_used >= NO_SEARCH_TURNS
                and self.search_cooldown == 0
                and self._post_catch_cooldown == 0):
            best_immediate = max(
                (CARPET_POINTS_TABLE.get(m.roll_length, 0)
                 for m in valid_moves if m.move_type == MoveType.CARPET),
                default=2,  # fallback: priming has ~2 pt future value
            )
            if search_ev > best_immediate:
                self.search_cooldown = SEARCH_COOLDOWN
                self.pos_history.append(my_pos)
                x, y = best_idx % 8, best_idx // 8
                return move.Move.search((x, y))

        # ── Forced search when cornered ────────────────────────────────────────
        if not valid_moves:
            self.search_cooldown = max(self.search_cooldown, SEARCH_COOLDOWN)
            self.pos_history.append(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        # ── Greedy carpet with patience ────────────────────────────────────────
        # Take immediately unless the run can be profitably extended and is safe.
        best_carpet = None
        best_carpet_pts = 0
        for m in valid_moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts > best_carpet_pts:
                    best_carpet_pts = pts
                    best_carpet = m
        if best_carpet_pts >= 4:
            can_ext, extra_pts = self._can_extend_run(board, my_pos, best_carpet)
            threatened = self._opponent_threatens_run(board, my_pos, best_carpet)
            # Take now if: late game, opponent can steal it, already 5+ len,
            # or extension gain is small enough not to justify waiting
            if (not can_ext
                    or threatened
                    or turns_left <= PATIENCE_MIN_TURNS
                    or best_carpet_pts >= 10
                    or extra_pts <= PATIENCE_GAIN_THRESHOLD):
                self.pos_history.append(my_pos)
                return best_carpet
            # Otherwise fall through — alpha-beta will decide between taking
            # now vs. extending the run over the next few turns

        # ── Iterative-deepening negamax ────────────────────────────────────────
        ordered = self._order_moves(valid_moves, 0)
        available = time_left() - TIME_BUFFER
        if available < 1.0:
            self.pos_history.append(my_pos)
            return ordered[0]

        turn_budget = min((available / max(turns_left, 1)) * 0.85, 6.0)
        deadline = time_left() - turn_budget

        best_move = ordered[0]
        for depth in range(2, 30):
            if time_left() < deadline + 0.05:
                break
            result = self._root_search(board, ordered, depth, time_left, deadline)
            if result is not None:
                best_move = result
                # PV move first for the next iteration
                ordered = [best_move] + [m for m in ordered if m is not best_move]
            else:
                break

        self.pos_history.append(my_pos)
        return best_move

    # ── Root search ────────────────────────────────────────────────────────────

    def _root_search(self, board, moves, depth, time_left, deadline):
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

            our_dest = child.player_worker.get_location()

            child.reverse_perspective()
            val = self._negamax(child, -beta, -alpha, depth - 1,
                                time_left, deadline, ply=1)
            if val is None:
                return None
            score = -val

            # Discourage revisiting recent positions (ping-pong prevention)
            if len(hist) >= 2 and our_dest == hist[-2]:
                score -= 50.0
            elif len(hist) >= 4 and our_dest == hist[-4]:
                score -= 25.0
            elif len(hist) >= 6 and our_dest == hist[-6]:
                score -= 12.0

            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)

        return best_move

    # ── Negamax with alpha-beta + transposition table ──────────────────────────

    def _negamax(self, board, alpha, beta, depth, time_left, deadline, ply):
        """
        Returns score from board.player_worker's perspective.
        Caller must reverse_perspective() the board before calling, and
        negate the returned value to convert back to the parent's frame.
        """
        if time_left() < deadline:
            return None

        if board.is_game_over() or depth == 0:
            return self._evaluate(board)

        # Transposition table probe
        key = self._board_key(board)
        tt = self._tt.get(key)
        if tt:
            d, flag, val = tt
            if d >= depth:
                if flag == _EXACT:
                    return val
                elif flag == _LOWER:
                    alpha = max(alpha, val)
                elif flag == _UPPER:
                    beta = min(beta, val)
                if alpha >= beta:
                    return val

        moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not moves:
            return self._evaluate(board)
        moves = self._order_moves(moves, ply)

        best = float('-inf')
        orig_alpha = alpha

        for m in moves:
            if time_left() < deadline:
                return None
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val = self._negamax(child, -beta, -alpha, depth - 1,
                                time_left, deadline, ply + 1)
            if val is None:
                return None
            val = -val

            if val > best:
                best = val
            alpha = max(alpha, best)
            if alpha >= beta:
                # Beta cutoff — record killer and update history
                self._update_killers(m, ply)
                mk = self._move_key(m)
                self._history[mk] = self._history.get(mk, 0) + depth * depth
                break

        # Store result in transposition table
        if len(self._tt) < MAX_TT_SIZE:
            if best <= orig_alpha:
                flag = _UPPER
            elif best >= beta:
                flag = _LOWER
            else:
                flag = _EXACT
            self._tt[key] = (depth, flag, best)

        return best

    # ── Move ordering ──────────────────────────────────────────────────────────

    def _move_key(self, m):
        if m.move_type == MoveType.CARPET:
            return (2, m.direction, m.roll_length)
        if m.move_type == MoveType.PRIME:
            return (1, m.direction, 0)
        return (0, m.direction, 0)

    def _update_killers(self, m, ply):
        if m.move_type == MoveType.CARPET:
            return  # Carpets are already ordered first; not useful as killers
        killers = self._killers.get(ply, [])
        mk = self._move_key(m)
        if killers and self._move_key(killers[0]) == mk:
            return
        self._killers[ply] = ([m] + killers)[:2]

    def _order_moves(self, moves, ply):
        """
        Order: carpets (best first) > killer quiet moves > primes > plains.
        Within each tier, history heuristic score breaks ties.
        """
        killers = self._killers.get(ply, [])
        killer_keys = {self._move_key(k) for k in killers}

        def priority(m):
            if m.move_type == MoveType.CARPET:
                return (0, -(CARPET_POINTS_TABLE.get(m.roll_length, 0)))
            mk = self._move_key(m)
            hist = self._history.get(mk, 0)
            if mk in killer_keys:
                return (1, -hist)
            if m.move_type == MoveType.PRIME:
                return (2, -hist)
            return (3, -hist)

        return sorted(moves, key=priority)

    # ── Carpet patience helpers ────────────────────────────────────────────────

    def _can_extend_run(self, board, pos, m):
        """
        Check whether carpet move m could be extended to a longer run.
        Returns (can_extend: bool, extra_pts: int) where extra_pts is the
        additional points gained by waiting for the best possible extension.
        """
        dx, dy = _DIR_DELTA[m.direction]
        end_x = pos[0] + dx * m.roll_length
        end_y = pos[1] + dy * m.roll_length
        next_x, next_y = end_x + dx, end_y + dy

        if not (0 <= next_x < 8 and 0 <= next_y < 8):
            return False, 0
        if board.get_cell((next_x, next_y)) != Cell.SPACE:
            return False, 0

        # Count consecutive extendable SPACE cells beyond the run end
        ext = 0
        nx, ny = next_x, next_y
        while 0 <= nx < 8 and 0 <= ny < 8 and board.get_cell((nx, ny)) == Cell.SPACE:
            ext += 1
            nx += dx
            ny += dy

        current_pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
        best_extra = 0
        for k in range(1, min(ext, 7 - m.roll_length) + 1):
            new_pts = CARPET_POINTS_TABLE.get(m.roll_length + k, 0)
            if new_pts - current_pts > best_extra:
                best_extra = new_pts - current_pts

        return best_extra > 0, best_extra

    def _opponent_threatens_run(self, board, pos, m):
        """
        True if the opponent is standing one step beyond the far end of the
        run — meaning they could carpet our primed cells in the reverse direction
        before we extend them.
        """
        dx, dy = _DIR_DELTA[m.direction]
        far_x = pos[0] + dx * m.roll_length + dx
        far_y = pos[1] + dy * m.roll_length + dy
        return board.opponent_worker.get_location() == (far_x, far_y)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def _evaluate(self, board):
        """
        Score from board.player_worker's perspective.
        Consistent with negamax: higher = better for the current mover.
        """
        my_pts  = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        turns_left = board.player_worker.turns_left
        my_pos  = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        late = turns_left <= 15
        score = (2.5 if late else 1.0) * float(my_pts - opp_pts)

        time_w = turns_left / 40.0 + 0.5  # scales heuristic terms with turns remaining

        # 1. Carpet portfolio: sum of all runs carpetable from each position
        my_cp  = self._carpet_portfolio(board, my_pos)
        opp_cp = self._carpet_portfolio(board, opp_pos)
        score += 2.5 * time_w * (my_cp - opp_cp)

        # 2. One-step reachable carpet: best run accessible by moving to a neighbour
        my_reach  = self._reachable_carpet(board, my_pos)
        opp_reach = self._reachable_carpet(board, opp_pos)
        score += 0.9 * time_w * (my_reach - opp_reach)

        # 3. Can-prime bonus: standing on SPACE means +1 available next turn.
        prime_w = 1.2 * time_w
        if board.get_cell(my_pos) == Cell.SPACE:
            score += prime_w
        if board.get_cell(opp_pos) == Cell.SPACE:
            score -= prime_w

        # 4. Mobility: reachable SPACE cells within 2 steps
        my_mob  = self._mobility(board, my_pos)
        opp_mob = self._mobility(board, opp_pos)
        score += 0.25 * (my_mob - opp_mob)

        # 5. Early-game run commitment (active for first ~14 turns, then fades)
        # Rewards positions with long unobstructed corridors to prime into.
        # Stops counting at BLOCKED, CARPET, opponent, or board edge so it
        # never steers us toward a dead-end or contested direction.
        # Fades exactly when _carpet_portfolio takes over as the dominant signal.
        early_w = max(0.0, (turns_left - 18) / 22.0)
        if early_w > 0.0:
            my_orp  = self._open_run_potential(board, my_pos)
            opp_orp = self._open_run_potential(board, opp_pos)
            score += 2.5 * early_w * (my_orp - opp_orp)

        return score

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _board_key(self, board):
        """Compact hashable representation of the current board state."""
        return (
            board._primed_mask,
            board._carpet_mask,
            board.player_worker.get_location(),
            board.opponent_worker.get_location(),
            board.player_worker.turns_left,
            board.opponent_worker.turns_left,
        )

    def _carpet_portfolio(self, board, pos):
        """
        Sum of carpet point values for every primed run reachable from pos
        (one step in each of the 4 cardinal directions).  Unlike v7.0's
        _best_carpet_value, this sums all directions instead of taking the max,
        giving a richer signal about total accessible run value.
        """
        total = 0.0
        x, y = pos
        for dx, dy in _DIRS4:
            run = 0
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                if board.get_cell((nx, ny)) == Cell.PRIMED:
                    run += 1
                    nx += dx
                    ny += dy
                else:
                    break
            # Only count runs of 2+ (length-1 scores -1 which penalises run-building)
            if run >= 2:
                total += CARPET_POINTS_TABLE.get(run, 0)
        return total

    def _reachable_carpet(self, board, pos):
        """
        Best carpet portfolio value reachable by taking ONE plain step.
        Captures runs that become accessible after a single movement move.
        """
        x, y = pos
        best = 0.0
        for dx, dy in _DIRS4:
            nx, ny = x + dx, y + dy
            if (0 <= nx < 8 and 0 <= ny < 8
                    and board.get_cell((nx, ny)) == Cell.SPACE):
                val = self._carpet_portfolio(board, (nx, ny))
                if val > best:
                    best = val
        return best

    def _mobility(self, board, pos):
        """Count SPACE cells reachable from pos within 2 BFS steps."""
        visited = {pos}
        frontier = [pos]
        count = 0
        for _ in range(2):
            nxt = []
            for cx, cy in frontier:
                for dx, dy in _DIRS4:
                    nb = (cx + dx, cy + dy)
                    if nb not in visited and 0 <= nb[0] < 8 and 0 <= nb[1] < 8:
                        if board.get_cell(nb) == Cell.SPACE:
                            visited.add(nb)
                            nxt.append(nb)
                            count += 1
            frontier = nxt
        return count

    def _open_run_potential(self, board, pos):
        """
        Best achievable carpet value from pos in any direction, counting
        consecutive non-blocked, non-carpet cells (SPACE or already PRIMED).

        Stops at: board edge, BLOCKED cell, CARPET cell, opponent position.
        This means the score naturally falls for:
          - Directions heading toward corners/walls  (fewer cells)
          - Directions where the opponent is sitting (they block the run)
          - Directions already cut off by carpet     (can't prime through carpet)

        Used only in the first ~14 turns; fades before _carpet_portfolio
        takes over once real run structures exist.
        """
        x, y = pos
        opp = board.opponent_worker.get_location()
        best = 0.0
        for dx, dy in _DIRS4:
            length = 0
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                if (nx, ny) == opp:
                    break
                cell = board.get_cell((nx, ny))
                if cell in (Cell.BLOCKED, Cell.CARPET):
                    break
                # SPACE and PRIMED both count toward the potential run
                length += 1
                nx += dx
                ny += dy
            if length >= 2:
                val = CARPET_POINTS_TABLE.get(min(length, 7), 0)
                if val > best:
                    best = val
        return best
