"""Download the three user-selected decks for local use; PNGs are gitignored."""

import argparse
from pathlib import Path
from urllib.request import urlopen

URL = "https://www.skatstube.de/images/redesign/cards/{deck}/g3@2x.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "skat/web/static/cards",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for deck in ("fr", "de", "by"):
        with urlopen(URL.format(deck=deck), timeout=30) as response:
            data = response.read()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{deck}: response is not a PNG")
        target = args.out / f"{deck}.png"
        temporary = target.with_suffix(".png.tmp")
        try:
            temporary.write_bytes(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded {target} (local only; do not commit)")


if __name__ == "__main__":
    main()
