#!/usr/bin/env python3
"""Register deepseek_v41 with transformers + vLLM, then hand off to the normal vllm CLI.

    python3 serve41.py serve /path/to/ckpt --tensor-parallel-size 4 ...

`vllm serve` offers no import hook, so out-of-tree registration must happen before its CLI
parses anything. Delegating to vllm.entrypoints.cli.main afterwards keeps all of vLLM's own
argument handling and multi-node plumbing.

THE SPLIT BETWEEN MODULE LEVEL AND __main__ IS LOAD-BEARING.

vLLM starts its engine core with multiprocessing `spawn`, and a spawned child RE-IMPORTS the
main module. So:

  * registration sits at MODULE level -- the spawned EngineCore child re-imports this file and
    must register the architecture too, or it cannot construct the model.
  * main() sits under `if __name__ == "__main__"` -- without the guard the child re-runs the
    whole server on import, which multiprocessing detects and rejects:
        RuntimeError: An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.
"""
import sys

import vllm_dsv41

# module level on purpose: spawned children re-import this file and need the registration
_REGISTERED = vllm_dsv41.register()

if __name__ == "__main__":
    print(f"[serve41] registered: {_REGISTERED}", flush=True)
    from vllm.entrypoints.cli.main import main  # import after registration

    sys.exit(main())
