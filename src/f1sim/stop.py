"""Stopping a long run cleanly: the first Ctrl+C (or SIGTERM, as a cluster's scheduler sends before a time limit) asks
the training to stop after the chunk it's in (it saves first), a second one stops at once."""

from __future__ import annotations

import signal


class StopFlag:
    def __init__(self):
        self.asked = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, sig, frame):
        if self.asked:
            raise KeyboardInterrupt
        self.asked = True
        print("\nstopping after this chunk and saving (Ctrl+C again to stop at once, without saving)", flush=True)
