"""
Helper for reporting missing optional dependencies with an actionable
``pip install graphguard[...]`` hint instead of a raw ImportError traceback.

GraphGuard's heavy dependencies (torch, fastapi, streamlit, gitpython, ...)
are optional extras (see pyproject.toml's ``[project.optional-dependencies]``)
so that ``graphguard analyze`` (parse + graph + features) works without
installing a GNN runtime, a REST framework, or a dashboard toolkit. Commands
that DO need one of those extras import it lazily inside the function body;
when the import fails, they should report a message built with this helper
rather than letting the raw ImportError propagate.
"""

from __future__ import annotations


def missing_dependency_message(extra: str, feature: str) -> str:
    """A one-line, actionable message for a missing optional dependency.

    Parameters
    ----------
    extra   : the pyproject.toml extra name that provides the dependency
              (e.g. "gnn", "serve", "dash", "git")
    feature : human-readable description of what needed it
              (e.g. "graphguard train", "the dashboard")
    """
    return (
        f"{feature} requires the '{extra}' extra, which isn't installed. "
        f"Install it with: pip install graphguard[{extra}]"
    )
