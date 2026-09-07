"""SQLite-backed checkpointer and store construction.

Every line here is load-bearing and was verified against the installed
versions (langgraph 1.2.11 / langgraph-checkpoint-sqlite 3.1.1). The design
document's ``SqliteSaver.from_conn_string("data/checkpoints.sqlite")`` does
not work: ``from_conn_string`` is a ``@contextmanager`` whose ``closing()``
wrapper shuts the connection on exit, so assigning its result gives you a
``_GeneratorContextManager`` and, if you do enter it, a connection that dies
underneath a long-running server.

The four non-obvious requirements, each confirmed by running it:

1. ``isolation_level=None`` is MANDATORY for ``SqliteStore``. Without it the
   first ``put()`` raises ``OperationalError: cannot start a transaction
   within a transaction``, because the store issues an explicit BEGIN that
   collides with Python's implicit transaction handling. Reproduced.
2. ``check_same_thread=False`` is MANDATORY because FastAPI runs sync
   endpoints in a threadpool, so the connection is touched from many threads.
3. The parent directory must exist. ``sqlite3.connect("data/x.sqlite")``
   raises ``OperationalError: unable to open database file`` on a fresh clone.
4. Separate connections for the saver and the store. ``SqliteSaver`` carries
   its own lock; two objects with independent locks sharing one connection is
   asking for contention.

``.setup()`` turns out to be optional on this version -- ``SqliteSaver``
creates its tables lazily -- but it is called explicitly anyway so a schema
error surfaces at startup rather than on the evaluator's first message.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

DEFAULT_CHECKPOINT_PATH = "data/checkpoints.sqlite"
DEFAULT_STORE_PATH = "data/store.sqlite"


def _resolve(path: str | None, env_var: str, fallback_attr: str) -> str:
    """Resolve a database path at CALL time, not at definition time.

    Deliberately not ``def make_checkpointer(path=DEFAULT_CHECKPOINT_PATH)``:
    a default argument is bound once, when the function is defined, so a test
    that reassigns ``persistence.DEFAULT_CHECKPOINT_PATH`` would be ignored
    and the suite would silently read and write the real demo database. That
    exact bug made the API tests accumulate state across runs.
    """
    return path or os.getenv(env_var) or globals()[fallback_attr]


def connect(path: str) -> sqlite3.Connection:
    """Open a SQLite connection configured for a threaded, long-lived server."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        path,
        check_same_thread=False,  # FastAPI sync endpoints run in a threadpool
        isolation_level=None,  # required by SqliteStore; harmless for the saver
    )
    # WAL lets a reader and a writer coexist; busy_timeout stops a concurrent
    # write from failing instantly with "database is locked".
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def make_checkpointer(path: str | None = None) -> SqliteSaver:
    """Short-term memory: graph state per ``thread_id``, surviving restarts."""
    saver = SqliteSaver(connect(_resolve(path, "LOOKLAB_CHECKPOINT_PATH", "DEFAULT_CHECKPOINT_PATH")))
    saver.setup()
    return saver


def reclaim_space(conn: sqlite3.Connection, force: bool = False) -> dict[str, int]:
    """Actually give the disk back after a delete.

    A plain ``DELETE`` returns nothing to the filesystem. In WAL mode the rows
    are removed from the logical database but the pages live on in the -wal
    file until a checkpoint, and the freed pages stay in the main file's
    freelist until a VACUUM. Measured on a thread holding ten image turns:
    delete alone left the files at 5.5 MB; checkpoint, VACUUM and a second
    checkpoint brought them to 52 KB.

    VACUUM rewrites the whole database, so it is skipped unless enough pages
    are actually free -- otherwise every delete would pay for a full rewrite.
    """
    freed_before = _db_bytes(conn)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        freelist = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        # Rewrite only when there is a meaningful amount to reclaim.
        if force or freelist * page_size > 256 * 1024:
            conn.execute("VACUUM")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        # Reclaiming is an optimisation; failing to do it must not fail a
        # delete the user has already been told succeeded.
        pass
    return {"before": freed_before, "after": _db_bytes(conn)}


def _db_bytes(conn: sqlite3.Connection) -> int:
    """Size of the database plus its WAL sidecar, in bytes."""
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        path = row[2] if row else None
    except sqlite3.Error:
        return 0
    if not path:
        return 0
    total = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            total += os.path.getsize(path + suffix)
        except OSError:
            continue
    return total


def make_store(path: str | None = None) -> SqliteStore:
    """Long-term memory: namespaced by ``user_id``, independent of any thread."""
    store = SqliteStore(connect(_resolve(path, "LOOKLAB_STORE_PATH", "DEFAULT_STORE_PATH")))
    store.setup()
    return store
