"""Refuse credential values inside AgentState (ADR-010)."""

_SECRET_KEYS = {"password", "secret", "api_key", "token", "authorization", "access_token"}


def reject_secret_values(value: object, path: str = "") -> None:
    """Raise when a serialized blob carries a credential.

    Keys that name a secret, and strings that look like a provider key,
    are rejected before the blob is written. Credential references (an id
    with no secret material) are ordinary strings and stay.
    """

    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            if name.lower() in _SECRET_KEYS and isinstance(item, str) and item:
                raise ValueError(f"secret value at {path}{name}")
            reject_secret_values(item, f"{path}{name}.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_secret_values(item, f"{path}{index}.")
        return
    if isinstance(value, str) and value.startswith("sk-"):
        raise ValueError(f"secret token at {path[:-1] or 'value'}")
