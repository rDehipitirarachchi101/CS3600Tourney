import numpy as np
from game.enums import Cell, Noise


class RatBelief:
    """
    Hidden Markov Model for tracking the rat's position.

    Belief vector b[i] = P(rat is at cell i).
    Updated each turn via:
        1. Prediction  : b' = b @ T  (rat moves)
        2. Observation : b''[i] ∝ b'[i] * P(noise|floor[i]) * P(est_dist|true_dist[i])
    """

    # ---------- noise likelihoods ----------
    # Assignment table columns are Squeak(0), Scratch(1), Squeal(2)
    # matching Noise enum: SQUEAK=0, SCRATCH=1, SQUEAL=2
    _P_NOISE = {
        Cell.BLOCKED: np.array([0.5, 0.3, 0.2], dtype=np.float32),
        Cell.SPACE:   np.array([0.7, 0.15, 0.15], dtype=np.float32),
        Cell.PRIMED:  np.array([0.1, 0.8, 0.1], dtype=np.float32),
        Cell.CARPET:  np.array([0.1, 0.1, 0.8], dtype=np.float32),
    }

    # ---------- distance-report likelihoods ----------
    # P(reported_dist | true_dist):  diff = reported - true
    _P_DIST = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

    def __init__(self, transition_matrix):
        self.t = np.array(transition_matrix, dtype=np.float32)  # (64, 64)
        self._init_belief()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, board, noise, est_d, w_pos, opp_search):
        """
        One full HMM update step.

        Parameters
        ----------
        board       : Board   current board (used for floor types)
        noise       : Noise   observed noise enum value
        est_d       : int     estimated manhattan distance from worker to rat
        w_pos       : (x, y)  worker's current position
        opp_search  : ((x,y), bool) | (None, False)  opponent's last search result
        """
        # 0. Check if a capture happened last turn (either player's correct guess)
        #    If so, a new rat was spawned — reset belief to post-burn-in prior.
        if (opp_search is not None and opp_search[0] is not None and opp_search[1]):
            self._init_belief()
            return  # fresh rat; no observation from this turn applies yet
        if (board.player_search[0] is not None and board.player_search[1]):
            self._init_belief()
            return

        # 1. Prediction: propagate belief through transition model
        self.b = self.b @ self.t

        # 2. Observation update
        n_idx = int(noise)
        wx, wy = w_pos

        # Build likelihood vector over all 64 cells (vectorised where possible)
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

        # 3. Zero-out cell where opponent just searched and missed
        if opp_search is not None and opp_search[0] is not None and not opp_search[1]:
            ox, oy = opp_search[0]
            new_b[oy * 8 + ox] = 0.0

        # 4. Zero-out cell where we just searched and missed
        if board.player_search[0] is not None and not board.player_search[1]:
            px, py = board.player_search[0]
            new_b[py * 8 + px] = 0.0

        # 5. Normalise — if all likelihoods collapsed keep prediction (don't reset to uniform)
        s = np.sum(new_b)
        if s > 1e-9:
            self.b = new_b / s
        # else: keep self.b from step 1 (prediction only; observation was uninformative)

    def get_best_guess(self):
        """Returns (flat_index, probability) of the most likely rat cell."""
        best_idx = int(np.argmax(self.b))
        return best_idx, float(self.b[best_idx])

    def search_ev(self, flat_idx):
        """Expected value of searching a specific cell."""
        p = float(self.b[flat_idx])
        return p * 4.0 - (1.0 - p) * 2.0  # = 6p - 2; positive when p > 1/3

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_belief(self):
        """Initialise (or reset) belief with 1000-step burn-in from (0,0)."""
        b = np.zeros(64, dtype=np.float32)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.t
        self.b = b