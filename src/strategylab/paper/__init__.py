"""Live paper trading of hall-of-fame bots.

Two commands (see __main__):

    python -m strategylab.paper select    # freeze the roster: top HOF bots per
                                          # pair that pass the hostile-futures
                                          # stress battery
    python -m strategylab.paper daemon    # live loop: refresh candles + deriv
                                          # metrics, replay each roster bot with
                                          # the real engine semantics, publish
                                          # reports/paper/state.json

The daemon is deliberately stateless: every cycle it re-derives positions and
the trade log by replaying each frozen genome from the paper epoch over the
live tape (swarm.trace mirrors the engine op-for-op). Restarts, crashes and
data refreshes are therefore always safe — there is no incremental state to
corrupt. The viewer's /paper page reads state.json; the server never
simulates anything (its standing contract).
"""
