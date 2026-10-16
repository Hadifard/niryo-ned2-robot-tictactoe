"""
File: niryo_tictactoe/common/game_logic.py
Description: Tic-Tac-Toe rules and agents (shared by all three vision variants).

Cells are indexed 0..8 and named p1..p9 (row-major, like the printed board):

    p1 p2 p3
    p4 p5 p6
    p7 p8 p9
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import random
from typing import Optional

EMPTY = " "
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]
# Preference order used to break ties between equally good moves.
PREFERENCE = [4, 0, 2, 6, 8, 1, 3, 5, 7]


def cell_name(index: int) -> str:
    return f"p{index + 1}"


def cell_index(name: str) -> int:
    return int(name[1:]) - 1


def other(player: str) -> str:
    return "O" if player == "X" else "X"


class TicTacToe:
    def __init__(self, board: Optional[list] = None):
        self.board = list(board) if board else [EMPTY] * 9

    def copy(self) -> "TicTacToe":
        return TicTacToe(self.board)

    def available_moves(self) -> list:
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def make_move(self, index: int, player: str) -> bool:
        if 0 <= index < 9 and self.board[index] == EMPTY:
            self.board[index] = player
            return True
        return False

    def winner(self) -> Optional[str]:
        for a, b, c in LINES:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_draw(self) -> bool:
        return self.winner() is None and not self.available_moves()

    def is_over(self) -> bool:
        return self.winner() is not None or not self.available_moves()

    def __str__(self) -> str:
        rows = [" | ".join(self.board[r * 3:r * 3 + 3]) for r in range(3)]
        return "\n---------\n".join(rows)


def board_from_colors(cells: list, color_to_player: dict) -> list:
    """Convert a vision reading (['blue','none',...]) to ['X',' ',...]."""
    return [color_to_player.get(c, EMPTY) for c in cells]


def diff_boards(previous: list, observed: list):
    """Compare the board we believe in with what the camera sees.

    Returns (new_moves, errors).  A clean human turn gives exactly one new move
    and no errors.  Anything else (a chip that vanished, changed colour, or two
    chips at once) is reported instead of being played - this is what prevents
    the repeated 'Invalid move' loop caused by a single bad camera frame.
    """
    new_moves, errors = [], []
    for i, (before, now) in enumerate(zip(previous, observed)):
        if before == now:
            continue
        if before == EMPTY:
            new_moves.append((i, now))
        elif now == EMPTY:
            errors.append(f"{cell_name(i)}: chip disappeared")
        else:
            errors.append(f"{cell_name(i)}: colour changed {before}->{now}")
    return new_moves, errors


# --------------------------------------------------------------------------- agents
def _minimax(board: list, to_move: str, me: str, alpha: float, beta: float, depth: int) -> int:
    game = TicTacToe(board)
    win = game.winner()
    if win == me:
        return 10 - depth          # prefer faster wins
    if win is not None:
        return depth - 10          # prefer slower losses
    if not game.available_moves():
        return 0
    if to_move == me:
        best = -99
        for move in game.available_moves():
            board[move] = to_move
            best = max(best, _minimax(board, other(to_move), me, alpha, beta, depth + 1))
            board[move] = EMPTY
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = 99
    for move in game.available_moves():
        board[move] = to_move
        best = min(best, _minimax(board, other(to_move), me, alpha, beta, depth + 1))
        board[move] = EMPTY
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


class MinimaxAgent:
    """Perfect player (alpha-beta, depth-aware). Never loses."""

    def __init__(self, player: str, rng: Optional[random.Random] = None):
        self.player, self.rng = player, rng

    def get_move(self, game: TicTacToe) -> int:
        board = list(game.board)
        scored = []
        for move in game.available_moves():
            board[move] = self.player
            scored.append((_minimax(board, other(self.player), self.player, -99, 99, 1), move))
            board[move] = EMPTY
        best_score = max(s for s, _ in scored)
        best = [m for s, m in scored if s == best_score]
        if self.rng:
            return self.rng.choice(best)
        return min(best, key=PREFERENCE.index)


class RandomAgent:
    def __init__(self, player: str, rng: Optional[random.Random] = None):
        self.player, self.rng = player, rng or random.Random()

    def get_move(self, game: TicTacToe) -> int:
        return self.rng.choice(game.available_moves())


class HybridAgent:
    """Minimax for its first `smart_moves` moves, then random (beatable on purpose)."""

    def __init__(self, player: str, smart_moves: int = 2, rng: Optional[random.Random] = None):
        self.smart = MinimaxAgent(player, rng)
        self.random = RandomAgent(player, rng)
        self.smart_moves_left = smart_moves

    def get_move(self, game: TicTacToe) -> int:
        if self.smart_moves_left > 0:
            self.smart_moves_left -= 1
            return self.smart.get_move(game)
        return self.random.get_move(game)


def make_agent(kind: str, player: str, rng: Optional[random.Random] = None):
    kind = kind.lower()
    if kind == "minimax":
        return MinimaxAgent(player, rng)
    if kind == "random":
        return RandomAgent(player, rng)
    if kind == "hybrid":
        return HybridAgent(player, rng=rng)
    raise ValueError(f"unknown agent '{kind}' (minimax | random | hybrid)")
