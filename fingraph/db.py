"""The Postgres database: LangGraph checkpoints and paragraph vectors live in the same one.

DATABASE_URL defaults to the docker-compose service (localhost:5433).
"""

import os
from contextlib import contextmanager

DEFAULT_URL = "postgresql://fingraph:fingraph@127.0.0.1:5433/fingraph"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def is_up(url: str | None = None) -> bool:
    import psycopg

    try:
        with psycopg.connect(url or database_url(), connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


@contextmanager
def checkpointer(url: str | None = None):
    """A LangGraph PostgresSaver, with its tables created if they are missing."""
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(url or database_url()) as saver:
        saver.setup()
        yield saver
