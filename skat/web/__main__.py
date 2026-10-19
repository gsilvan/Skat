"""Run with: uv run --extra web python -m skat.web."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Lokales Skat-Webinterface")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    try:
        from .app import create_app
    except ModuleNotFoundError as error:
        if error.name == "flask":
            parser.exit(1, "Flask fehlt: uv sync --extra web\n")
        raise
    app = create_app()
    try:
        app.run(
            host="127.0.0.1",
            port=args.port,
            threaded=True,
            use_reloader=False,
            debug=False,
        )
    finally:
        app.extensions["skat_sessions"].close()


if __name__ == "__main__":
    main()
