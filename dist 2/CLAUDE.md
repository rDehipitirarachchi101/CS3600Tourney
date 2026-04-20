# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS3600 competitive AI tournament — a two-player carpet-rolling board game on an 8x8 grid. Players prime cells, carpet over them to score points, and hunt an invisible rat using noisy sensor data. Each player has 240 seconds and 40 turns.

## Running Matches

```bash
# Run a match between two agents (from the repo root)
python3 engine/run_local_agents.py <player_a_name> <player_b_name>

# Example
python3 engine/run_local_agents.py my_bot my_bot_albert

# Analyze recorded match results
python3 analyze_matches.py
```

Agent names correspond to folder names under `3600-agents/`. Match results are saved as JSON to `3600-agents/matches/`.

## Agent Interface

Every agent lives in `3600-agents/<name>/` and must export `PlayerAgent` and `rat_belief` via `__init__.py`.

**`agent.py` — required class:**
```python
class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None): ...
    def play(self, board, sensor_data: Tuple, time_left: Callable) -> Move: ...
    def commentate(self) -> str: ...  # optional, called at game end
```

**`rat_belief.py` — required class:**
```python
class RatBelief:
    def update(self, board, noise, est_d, w_pos, opp_search): ...
    def get_best_guess(self) -> tuple[int, float]: ...  # (cell_idx, probability)
    def search_ev(self, flat_idx: int) -> float: ...
```

## Board Perspective

Agents always see themselves as `board.player` and the opponent as `board.enemy`. After each turn the engine calls `board.reverse_perspective()`, so agent code never needs to track "am I A or B".

## Key Game Mechanics

**Scoring:**
- Prime a cell: +1 pt
- Carpet run of length N: 2→+2, 3→+4, 4→+6, 5→+10, 6→+15, 7→+21
- Catch rat (successful search): +4 pts
- Failed search: −2 pts

**Move types** (`enums.MoveType`): `PLAIN`, `PRIME`, `CARPET`, `SEARCH`

**Rat sensor data** — each turn `play()` receives `(noise, estimated_distance)`:
- Noise: `SQUEAK` (rat on SPACE), `SCRATCH` (rat on PRIMED), `SQUEAL` (rat on CARPET)
- Distance: true distance ± error drawn from `[0.12, 0.70, 0.12, 0.06]` for errors `[±2, ±1, 0, ±2]`

## Architecture

```
engine/
  game/
    board.py          # Game state — bitboards, apply_move(), forecast_move(), get_valid_moves()
    enums.py          # MoveType, Cell, Noise, CARPET_POINTS_TABLE
    move.py           # Move factory: Move.plain(), .prime(), .carpet(), .search()
    rat.py            # Hidden rat — transition matrix + noisy sensor sampling
    worker.py         # Per-player state: position, points, turns_left, time_left
    history.py        # Replay recording
  gameplay.py         # play_game() — main game loop (533 lines)
  player_process.py   # Subprocess sandboxing with timeouts + memory limits
  run_local_agents.py # CLI entry point
  board_utils.py      # Terminal display + JSON serialization
  transition_matrices/ # Pickle files (64×64 rat movement probabilities)

3600-agents/
  my_bot/             # Current submission — AKIR-v7.0 (alpha-beta)
  my_bot_mcts/        # MCTS variant
  my_bot_albert/      # Expectiminimax variant (simplified, no rat search)
  OldBots/            # Archived versions
  matches/            # Recorded match JSON files
```

## Current Agent Strategy (my_bot — AKIR-v7.0)

- **Search**: Alpha-beta minimax with iterative deepening; move ordering carpet > prime > plain
- **Rat hunting**: HMM belief state over all 64 cells; only searches when `EV > opportunity_cost` (where priming = +1)
- **Key parameters**: `SEARCH_EV_MIN=1.0`, `SEARCH_COOLDOWN=4`, `NO_SEARCH_TURNS=8`, `TIME_BUFFER=5.0`
- **Late-game weight**: point differential weighted 2.5× in evaluation during last ~10 turns
- **Oscillation penalty**: penalizes revisiting recent positions to prevent ping-ponging

## Time Budget

Agents have ~6 s/turn on average (240 s / 40 turns). The engine kills agents that exceed their wall-clock budget. Implement anytime search — commit to the best move found so far and check `time_left()` frequently.
