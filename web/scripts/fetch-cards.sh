#!/usr/bin/env sh
set -eu

target="$(dirname "$0")/../public/cards/skatstube-g1@2x.png"
mkdir -p "$(dirname "$target")"
curl -fL "https://www.skatstube.de/images/redesign/cards/fr/g1@2x.png" -o "$target"
echo "Downloaded local-only sprite to $target (ignored by Git)."
