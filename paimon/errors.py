"""The root of everything Paimon raises on purpose.

Paimon is embeddable as a library, so a caller needs one way to tell "the
agent refused, and here is why" from a bug that happened to surface as a
``ValueError``. Every deliberate failure inherits from ``PaimonError``; the
concrete classes stay in the module that raises them.

Not a replacement for the builtins. A malformed argument is still a
``ValueError`` (``split_model_string``, ``validate_profile``) because that is
what it is. What lives under this root is a refusal about state: a session
already open, a job that cannot start, an answer that never came.
"""


class PaimonError(Exception):
    """Anything Paimon raises deliberately."""
