"""Refresh the owned session's proxy credentials using the Colab CLI client.

Run with the CLI's Python interpreter. The installed CLI's keep-alive maintains
the VM but does not update its cached, expiring runtime proxy credentials.
"""

import json
import sys


def refresh(state, name, endpoint):
    session = state.store.get(name)
    if session is None or session.endpoint != endpoint:
        raise RuntimeError("Owned Colab session is missing or its endpoint changed")
    assignment = next((item for item in state.client.list_assignments()
                       if item.endpoint == endpoint), None)
    if assignment is None:
        raise RuntimeError("Owned Colab runtime is no longer assigned")
    proxy = assignment.runtime_proxy_info
    # Update credentials only for the existing VM; keep its kernel and session
    # identity so refreshing cannot provision or substitute a different run.
    session.token, session.url = proxy.token, proxy.url
    state.store.add(session)
    return {"expires_in_seconds": proxy.token_expires_in_seconds}


if __name__ == "__main__":
    from colab_cli.common import state

    print(json.dumps(refresh(state, *sys.argv[1:])))
