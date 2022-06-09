#!/usr/bin/env python3

import os

import requests
from bs4 import BeautifulSoup

username = "change_me"
password = "change_me"

s = requests.Session()
s.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/95.0.4638.69 Safari/537.36",
        "accept-encoding": "gzip",
    }
)

resp = s.get("https://www.skatstube.de/")
soup = BeautifulSoup(resp.content, features="html.parser")
a_token = soup.select_one("input[name='authenticity_token']")["value"]

login_data = {
    "login": username,
    "password": password,
    "authenticity_token": a_token,
    "remember_me": "1",
    "utf8": "✓",
}
r = s.post("https://www.skatstube.de/login", data=login_data)

for game_idx in range(1003757069, 1165216708):
    if not os.path.exists(f"games/{game_idx}.json"):
        r = s.get(f"https://www.skatstube.de/spiele/{game_idx}.json")
        with open(f"games/{game_idx}.json", "wb") as f:
            f.write(r.content)
    else:
        print(f"skipping game {game_idx}")
