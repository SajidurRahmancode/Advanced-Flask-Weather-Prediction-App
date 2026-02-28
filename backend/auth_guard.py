"""
auth_guard.py — Shared authentication decorator for API routes.

Replaces the 21 inline session checks previously copy-pasted across routes.py.
Usage:
    from backend.auth_guard import require_auth, get_session_user_id

    @api.route('/weather/predict', methods=['POST'])
    @require_auth
    def predict():
        user_id = get_session_user_id()
        ...
"""

import logging
from functools import wraps

from flask import session, jsonify

logger = logging.getLogger(__name__)


def require_auth(f):
    """
    Decorator that enforces session-based authentication.

    Returns 401 JSON if 'user_id' is absent from the session.
    Identical behaviour to the 21 inline checks it replaces, but
    defined once and testable in isolation.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            logger.debug(
                "Unauthenticated request to %s", f.__name__
            )
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated


def get_session_user_id() -> int:
    """
    Return the current user's id from the session.

    Safe to call only inside a view decorated with @require_auth.
    """
    return session['user_id']
