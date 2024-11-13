import random
import re
import hashlib


def parse_text(text: str):
    while True:
        match = re.search(r"\{([^\{\}]*)\}", text)
        if match is None:
            break
        options = match.group(1).split("|")
        choice = random.choice(options)
        text = text.replace(match.group(0), choice, 1)
    s = text.split(",")
    s = map(str.strip, s)
    s = filter(bool, s)
    text = ", ".join(s)
    return text


def hash_var(var: str):
    sha256 = hashlib.sha256()
    sha256.update(var.encode())
    hash_value = sha256.hexdigest()[:10]
    return hash_value


def log(msg):
    print(f"[SQNodes] {msg}")


if __name__ == "__main__":
    text = "chicken, poo, {cat||}, {dog|dead {fish|whale}}"
    print(parse_text(text))
