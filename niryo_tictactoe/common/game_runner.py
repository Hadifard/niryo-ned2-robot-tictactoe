"""
File: niryo_tictactoe/common/game_runner.py
Description: Game loop shared by all variants.

A *backend* hides how positions are found; the three variants only differ here:

    backend.read_board()   -> (cells[9], token_side)  or None while the view is unstable
    backend.cell_xy(i)     -> robot XY (m) of cell i
    backend.slot_xy(name)  -> robot XY (m) of a chip in tray slot 'c1'..'c10'
    backend.token_xy(side) -> robot XY (m) of the turn-token spot 'A' / 'B'

Turn protocol (same idea as the original project): the red token lies on the
spot of the side that must play.  After my move I carry it to the opponent's
spot; the opponent (human or the other robot) puts it back on mine when done.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from common.game_logic import (EMPTY, TicTacToe, board_from_colors, cell_name,
                               diff_boards, make_agent)


class GameRunner:
    def __init__(self, arm, backend, me_color: str, opp_color: str, agent_kind: str,
                 my_side: str, opp_side: str, my_slots: list, i_start: bool,
                 poll: float = 0.3, log: Callable = print, timeout: float = 300.0,
                 sleep: Callable = time.sleep):
        self.arm, self.backend = arm, backend
        self.me_color, self.opp_color = me_color, opp_color
        self.my_side, self.opp_side = my_side, opp_side
        self.my_slots, self.i_start = list(my_slots), i_start
        self.poll, self.log, self.timeout, self.sleep = poll, log, timeout, sleep
        self.me, self.opp = "O", "X"
        self.agent = make_agent(agent_kind, self.me)
        self.colors = {me_color: self.me, opp_color: self.opp}
        self.game = TicTacToe()
        self.my_cells: list = []

    # ------------------------------------------------------------------ helpers
    def _observe(self):
        reading = self.backend.read_board()
        if reading is None:
            return None
        cells, token = reading
        return board_from_colors(cells, self.colors), token

    def _wait_until(self, predicate, what: str):
        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            observed = self._observe()
            if observed is not None:
                result = predicate(*observed)
                if result is not None:
                    return result
            self.sleep(self.poll)
        raise TimeoutError(f"timeout while waiting for {what}")

    # ------------------------------------------------------------------- turns
    def _standby_until_token(self):
        """REST pose (folded, cannot be hit) and watch only the red-token region."""
        self.arm.go_rest()
        watch = getattr(self.backend, "token_at_rest", None)
        if watch is None:
            def watch(side):
                reading = self.backend.read_board()
                return reading is not None and reading[1] == side
        start = time.monotonic()
        while not watch(self.my_side):
            if time.monotonic() - start > self.timeout:
                raise TimeoutError("timeout while waiting for my turn token")
            self.sleep(self.poll)

    def my_move(self):
        self.arm.go_observe()                          # fixed observation point
        self._wait_until(lambda board, token: True, "a stable camera view")
        move = self.agent.get_move(self.game)
        slot = self.my_slots[len(self.my_cells)]
        self.log(f"robot plays {cell_name(move)} (chip from {slot})")
        self.arm.pick_and_place(self.backend.slot_xy(slot), self.backend.cell_xy(move),
                                slot, cell_name(move))
        self.game.make_move(move, self.me)
        self.my_cells.append(move)
        self.arm.pick_and_place(self.backend.token_xy(self.my_side),
                                self.backend.token_xy(self.opp_side), "token", "token")
        self.arm.go_rest()                             # standby while the opponent plays

    def opponent_move(self):
        self._standby_until_token()          # token back on my side = opponent finished
        self.arm.go_observe()
        bad_frames = 0

        def check(board, token):
            nonlocal bad_frames
            if token != self.my_side:
                return None
            new, errors = diff_boards(self.game.board, board)
            if len(new) == 1 and not errors and new[0][1] == self.opp:
                return new[0][0]
            bad_frames += 1
            if bad_frames % 10 == 0:
                self.log(f"unclear board ({new}, {errors}) - waiting")
            return None

        index = self._wait_until(check, "opponent move")
        self.game.make_move(index, self.opp)
        self.log(f"opponent played {cell_name(index)}")

    # -------------------------------------------------------------------- game
    def run(self) -> Optional[str]:
        my_turn = self.i_start
        while not self.game.is_over():
            self.my_move() if my_turn else self.opponent_move()
            my_turn = not my_turn
        winner = self.game.winner()
        self.log({"O": "robot wins", "X": "opponent wins", None: "draw"}[winner])
        return winner

    def clear_board(self, tray_slots: Optional[list] = None):
        """Put my chips back into the tray (the opponent clears its own)."""
        for cell, slot in zip(self.my_cells, tray_slots or self.my_slots):
            self.arm.pick_and_place(self.backend.cell_xy(cell), self.backend.slot_xy(slot),
                                    cell_name(cell), slot)
