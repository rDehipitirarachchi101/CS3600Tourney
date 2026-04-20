import math
import random
from collections.abc import Callable
from typing import List, Tuple, Optional
from collections import deque

from game import board, move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, MoveType, Direction, loc_after_direction
from .rat_belief import RatBelief


TIME_BUFFER = 8.0       # extra safe for MCTS
HISTORY_LEN = 6
UCB_C = 1.41            # exploration constant
MAX_ROLLOUT_DEPTH = 12  # how many moves to simulate per rollout
_FOUR_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class MCTSNode:
    __slots__ = ('board_state', 'parent', 'move', 'children',
                 'visits', 'total_value', 'untried_moves', 'is_max')

    def __init__(self, board_state, parent=None, move=None, is_max=True):
        self.board_state = board_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_moves = None  # lazy init
        self.is_max = is_max

    def ucb_score(self, parent_visits):
        if self.visits == 0:
            return float('inf')
        exploit = self.total_value / self.visits
        explore = UCB_C * math.sqrt(math.log(parent_visits) / self.visits)
        if self.is_max:
            return exploit + explore
        else:
            return -exploit + explore  # opponent wants to minimize our score

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb_score(self.visits))

    def best_move_child(self):
        """After search, pick the most-visited child."""
        return max(self.children, key=lambda c: c.visits)


class PlayerAgent:
    """
    MCTS bot for the carpet game.
    
    Each turn:
    1. Handle rat search (same as v7.0 — only when very confident)
    2. Greedy carpet if length 3+ available
    3. Run MCTS for remaining time budget
    4. Pick the most-visited root action
    
    Rollout policy: random but biased toward carpet > prime > plain.
    Evaluation at rollout end: point differential + carpet potential.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = RatBelief(transition_matrix)
        self.pos_history = deque(maxlen=HISTORY_LEN)
        self.search_cooldown = 0

    def commentate(self):
        return "AKIR-MCTS-v1"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, est_d = sensor_data
        my_pos = board.player_worker.get_location()
        self.hmm.update(board, noise, est_d, my_pos, board.opponent_search)
        self.search_cooldown = max(0, self.search_cooldown - 1)

        best_idx, _ = self.hmm.get_best_guess()
        turns_left = board.player_worker.turns_left
        turns_used = 40 - turns_left

        # Rat search — same as v7.0 (only when very confident)
        search_ev = self.hmm.search_ev(best_idx)
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)

        if (search_ev > 1.0
                and turns_used >= 10
                and self.search_cooldown == 0):
            best_carpet_pts = 0
            for m in valid_moves:
                if m.move_type == MoveType.CARPET:
                    pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                    if pts > best_carpet_pts:
                        best_carpet_pts = pts
            if search_ev > best_carpet_pts:
                self.search_cooldown = 4
                self.pos_history.append(my_pos)
                x, y = best_idx % 8, best_idx // 8
                return move.Move.search((x, y))

        # No valid moves → forced search
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if not valid_moves:
            best_idx, _ = self.hmm.get_best_guess()
            self.pos_history.append(my_pos)
            x, y = best_idx % 8, best_idx // 8
            return move.Move.search((x, y))

        # Greedy: take best carpet if length 3+
        best_carpet = None
        best_carpet_pts = 0
        for m in valid_moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts > best_carpet_pts:
                    best_carpet_pts = pts
                    best_carpet = m
        if best_carpet_pts >= 4:
            self.pos_history.append(my_pos)
            return best_carpet

        # Run MCTS
        available = time_left() - TIME_BUFFER
        if available < 1.0:
            self.pos_history.append(my_pos)
            return self._pick_default(valid_moves)

        time_per_turn = available / max(turns_left, 1)
        budget = min(time_per_turn * 0.85, 5.0)
        deadline = time_left() - budget

        best_move = self._mcts(board, valid_moves, time_left, deadline)

        self.pos_history.append(my_pos)
        return best_move

    # ------------------------------------------------------------------
    # MCTS
    # ------------------------------------------------------------------

    def _mcts(self, board, valid_moves, time_left, deadline):
        root = MCTSNode(board, is_max=True)
        root.untried_moves = list(valid_moves)

        iterations = 0
        while time_left() > deadline:
            # 1. Selection
            node = root
            sim_board = board.get_copy()

            while node.untried_moves is not None and len(node.untried_moves) == 0 and node.children:
                node = node.best_child()
                # Apply the move
                if node.is_max:
                    # This was an opponent move, need to reverse first
                    sim_board.reverse_perspective()
                sim_board.apply_move(node.move, check_ok=False)
                if not node.is_max:
                    sim_board.reverse_perspective()

            # 2. Expansion
            if node.untried_moves is None:
                if node.is_max:
                    node.untried_moves = sim_board.get_valid_moves(enemy=False, exclude_search=True)
                else:
                    sim_board.reverse_perspective()
                    node.untried_moves = sim_board.get_valid_moves(enemy=False, exclude_search=True)
                    sim_board.reverse_perspective()

            if node.untried_moves:
                m = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
                child_is_max = not node.is_max

                if node.is_max:
                    child_board = sim_board.forecast_move(m, check_ok=False)
                else:
                    sim_board.reverse_perspective()
                    child_board = sim_board.forecast_move(m, check_ok=False)
                    if child_board:
                        child_board.reverse_perspective()
                    sim_board.reverse_perspective()

                if child_board is None:
                    continue

                child = MCTSNode(child_board, parent=node, move=m, is_max=child_is_max)
                node.children.append(child)
                node = child
                sim_board = child_board

            # 3. Simulation (rollout)
            value = self._rollout(sim_board, node.is_max)

            # 4. Backpropagation
            while node is not None:
                node.visits += 1
                node.total_value += value
                node = node.parent

            iterations += 1

        # Pick most visited child of root
        if not root.children:
            return self._pick_default(valid_moves)

        best = root.best_move_child()
        return best.move

    def _rollout(self, board, is_max_turn):
        """
        Fast random rollout. Returns score from our perspective.
        Biased random: prefer carpet > prime > plain.
        """
        sim = board.get_copy()

        for _ in range(MAX_ROLLOUT_DEPTH):
            if is_max_turn:
                moves = sim.get_valid_moves(enemy=False, exclude_search=True)
            else:
                sim.reverse_perspective()
                moves = sim.get_valid_moves(enemy=False, exclude_search=True)

            if not moves:
                if not is_max_turn:
                    sim.reverse_perspective()
                break

            # Biased selection
            m = self._biased_random_move(moves)

            sim.apply_move(m, check_ok=False)
            if not is_max_turn:
                sim.reverse_perspective()

            is_max_turn = not is_max_turn

        # Evaluate from our perspective
        return self._rollout_eval(sim, is_max_turn)

    def _biased_random_move(self, moves):
        """Pick a move with bias: carpet(long) > prime > plain > carpet(1)."""
        carpets = []
        primes = []
        plains = []
        for m in moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts > 0:
                    carpets.append((pts, m))
                else:
                    plains.append(m)  # length-1 carpet treated as plain
            elif m.move_type == MoveType.PRIME:
                primes.append(m)
            else:
                plains.append(m)

        # 50% chance to pick best carpet if available
        if carpets and random.random() < 0.5:
            carpets.sort(key=lambda x: -x[0])
            return carpets[0][1]
        # 30% chance to pick a prime
        if primes and random.random() < 0.6:
            return random.choice(primes)
        # Otherwise random from all
        return random.choice(moves)

    def _rollout_eval(self, board, is_max_turn):
        """Simple evaluation at end of rollout."""
        # Make sure we're evaluating from the right perspective
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()

        if not is_max_turn:
            # We need to swap since perspective might be flipped
            my_pts, opp_pts = opp_pts, my_pts

        score = float(my_pts - opp_pts)

        # Small carpet potential bonus
        my_pos = board.player_worker.get_location()
        if is_max_turn:
            score += 0.5 * self._carpet_potential(board, my_pos)

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

    def _pick_default(self, moves):
        """Fallback: pick best carpet, then any prime, then any plain."""
        best = None
        best_score = -999
        for m in moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts > best_score:
                    best_score = pts
                    best = m
            elif m.move_type == MoveType.PRIME and best_score < 1:
                best_score = 1
                best = m
            elif best is None:
                best = m
        return best
