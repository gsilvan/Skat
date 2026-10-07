from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_sock import Sock

from skat.agents.bidding import BiddingStyle

from .session import AGENT_KINDS, SessionRegistry, discover_checkpoints


def create_app(*, registry: SessionRegistry | None = None) -> Flask:
    root = Path(__file__).resolve().parents[2]
    frontend = root / "web" / "dist"
    app = Flask(__name__, static_folder=None)
    sock = Sock(app)
    sessions = registry or SessionRegistry()
    app.extensions["skat_sessions"] = sessions

    @app.errorhandler(ValueError)
    def invalid_request(error: ValueError):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(TypeError)
    def invalid_type(error: TypeError):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(KeyError)
    def missing_session(error: KeyError):
        return jsonify({"error": str(error)}), 404

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/config")
    def config():
        checkpoints = discover_checkpoints(root)
        return jsonify(
            {
                "agents": list(AGENT_KINDS),
                "biddingStyles": [style.value for style in BiddingStyle],
                "checkpoints": checkpoints,
                "spriteAvailable": (
                    root / "web" / "public" / "cards" / "skatstube-g1@2x.png"
                ).exists(),
            }
        )

    @app.post("/api/sessions")
    def create_session():
        body = request.get_json(silent=False)
        session = sessions.create(body)
        return jsonify(session.get_snapshot()), 201

    @app.get("/api/sessions/<session_id>")
    def get_session(session_id: str):
        return jsonify(sessions.get(session_id).get_snapshot())

    @app.post("/api/sessions/<session_id>/actions")
    def submit_action(session_id: str):
        body = request.get_json(silent=False)
        if not isinstance(body, dict) or "value" not in body:
            raise ValueError("an action requires a value")
        session = sessions.get(session_id)
        session.submit_action(body["value"])
        return jsonify({"accepted": True})

    @app.post("/api/sessions/<session_id>/control")
    def control_session(session_id: str):
        body = request.get_json(silent=False)
        if not isinstance(body, dict) or "command" not in body:
            raise ValueError("a control request requires a command")
        session = sessions.get(session_id)
        session.control(str(body["command"]), body.get("value"))
        return jsonify(session.get_snapshot())

    @app.delete("/api/sessions/<session_id>")
    def stop_session(session_id: str):
        sessions.get(session_id).stop()
        return "", 204

    @sock.route("/api/sessions/<session_id>/events")
    def session_events(ws: Any, session_id: str):
        session = sessions.get(session_id)
        state = session.get_snapshot()
        while True:
            ws.send(json.dumps(state))
            if state["status"] in {"finished", "passed", "error", "stopped"}:
                return
            state = session.wait_for_update(int(state["revision"]))

    @app.get("/")
    def index():
        index_file = frontend / "index.html"
        if index_file.exists():
            return send_from_directory(frontend, "index.html")
        return (
            "<h1>Skat Web GUI</h1>"
            "<p>The React bundle is missing. Run <code>cd web && npm install "
            "&& npm run build</code>, or use <code>npm run dev</code>.</p>"
        )

    @app.get("/<path:path>")
    def frontend_asset(path: str):
        candidate = frontend / path
        if candidate.is_file():
            return send_from_directory(frontend, path)
        return send_from_directory(frontend, "index.html")

    return app
