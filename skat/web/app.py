"""Same-origin HTTP interface to local engine sessions."""

import atexit
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .session import KINDS, Conflict, SessionRegistry, discover_checkpoints


def create_app(*, registry=None):
    app = Flask(__name__, static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    sessions = registry if registry is not None else SessionRegistry()
    app.extensions["skat_sessions"] = sessions
    atexit.register(sessions.close)

    @app.before_request
    def same_origin():
        if request.method in {"POST", "DELETE"}:
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                return jsonify(error="Fremder Ursprung ist nicht erlaubt"), 403
            if request.headers.get("Sec-Fetch-Site") == "cross-site":
                return jsonify(error="Fremder Ursprung ist nicht erlaubt"), 403
        # Prevent DNS rebinding of a server intentionally bound to loopback.
        if urlsplit(request.host_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            return jsonify(error="Nur lokale Hostnamen sind erlaubt"), 403

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = (
            "no-store" if request.path.startswith("/api/") else "no-cache"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.errorhandler(Conflict)
    def conflict(error):
        return jsonify(error=str(error)), 409

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(KeyError)
    def missing(error):
        return jsonify(error="Partie nicht gefunden"), 404

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    def body():
        value = request.get_json()
        if not isinstance(value, dict):
            raise ValueError("JSON-Objekt erforderlich")
        return value

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/config")
    def config():
        root = Path(__file__).resolve().parents[2]
        return jsonify(
            agents=KINDS,
            checkpoints=discover_checkpoints(root),
            decks={
                name: (Path(app.static_folder) / "cards" / f"{name}.png").is_file()
                for name in ("fr", "de", "by")
            },
        )

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.post("/api/sessions")
    def create():
        return jsonify(sessions.create(body()).get_snapshot()), 201

    @app.get("/api/sessions/<key>")
    def state(key):
        after = request.args.get("after")
        if after is not None:
            try:
                after = int(after)
            except ValueError:
                raise ValueError("Ungültige Revision") from None
        return jsonify(sessions.get(key).get_snapshot(after=after))

    @app.post("/api/sessions/<key>/actions")
    def action(key):
        value = body()
        if "decisionId" not in value or "value" not in value:
            raise ValueError("decisionId und value erforderlich")
        sessions.get(key).submit(value["decisionId"], value["value"])
        return jsonify(accepted=True)

    @app.post("/api/sessions/<key>/control")
    def control(key):
        value = body()
        sessions.get(key).control(
            value.get("command"), value.get("value"), value.get("roundIndex")
        )
        return jsonify(accepted=True)

    @app.delete("/api/sessions/<key>")
    def delete(key):
        sessions.remove(key)
        return "", 204

    return app
