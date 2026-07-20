"""CLI dispatcher: python -m strategylab.paper {select|daemon} [args...]"""

from __future__ import annotations

import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    argv = sys.argv[2:]
    if cmd == "select":
        from .select import main as run
    elif cmd == "daemon":
        from .daemon import main as run
    else:
        sys.exit("usage: python -m strategylab.paper {select|daemon} [options]\n"
                 "  select  freeze the roster (top stress-passing HOF bots)\n"
                 "  daemon  run the live paper-trading loop")
    run(argv)


if __name__ == "__main__":
    main()
