import numpy as np
from game.enums import Cell, Noise


class RatBelief:
    """
    Hidden Markov Model for tracking the rat's position.
    Belief vector b[i] = P(rat is at cell i).
    """

    _P_NOISE = {
        Cell.BLOCKED: np.array([0.5, 0.3, 0.2], dtype=np.float32),
        Cell.SPACE:   np.array([0.7, 0.15, 0.15], dtype=np.float32),
        Cell.PRIMED:  np.array([0.1, 0.8, 0.1], dtype=np.float32),
        Cell.CARPET:  np.array([0.1, 0.1, 0.8], dtype=np.float32),
    }

    # P(est_d - true_d = offset)
    _P_DIST = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

    def __init__(self, transition_matrix):
        self.t = np.array(transition_matrix, dtype=np.float32)
        self._init_belief()

    def update(self, board, noise, est_d, w_pos, opp_search):
        # Rat was caught — reset to stationary distribution
        if opp_search is not None and opp_search[0] is not None and opp_search[1]:
            self._init_belief()
            return
        if board.player_search[0] is not None and board.player_search[1]:
            self._init_belief()
            return

        # Transition step
        self.b = self.b @ self.t

        # Observation step
        n_idx = int(noise)
        wx, wy = w_pos
        likelihood = np.zeros(64, dtype=np.float32)

        for i in range(64):
            if self.b[i] < 1e-6:
                continue
            x, y = i % 8, i // 8
            floor = board.get_cell((x, y))
            l_noise = self._P_NOISE.get(floor, self._P_NOISE[Cell.SPACE])[n_idx]
            true_d = abs(x - wx) + abs(y - wy)
            d_diff = est_d - true_d
            l_dist = self._P_DIST.get(d_diff, 0.0)
            likelihood[i] = l_noise * l_dist

        new_b = self.b * likelihood

        # Zero out searched cells
        if opp_search is not None and opp_search[0] is not None and not opp_search[1]:
            ox, oy = opp_search[0]
            new_b[oy * 8 + ox] = 0.0

        if board.player_search[0] is not None and not board.player_search[1]:
            px, py = board.player_search[0]
            new_b[py * 8 + px] = 0.0

        s = np.sum(new_b)
        if s > 1e-9:
            self.b = new_b / s

    def get_best_guess(self):
        best_idx = int(np.argmax(self.b))
        return best_idx, float(self.b[best_idx])

    def search_ev(self, flat_idx):
        p = float(self.b[flat_idx])
        return p * 4.0 - (1.0 - p) * 2.0

    def _init_belief(self):
        b = np.zeros(64, dtype=np.float32)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.t
        self.b = b
