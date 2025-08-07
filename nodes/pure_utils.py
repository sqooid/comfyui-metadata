import random
import re
import hashlib

import numpy


def parse_text(text: str):
    while True:
        match = re.search(r"\{([^\{\}]*)\}", text)
        if match is None:
            break
        options = match.group(1).split("|")
        weights = []
        for o in options:
            weight_match = re.search(r"\s*^(\d*\.?\d*):", o)
            if weight_match:
                weight = float(weight_match.group(1))
                o = o.replace(weight_match.group(0), "")
                weights.append(weight)
            else:
                weights.append(1)
        log(weights)
        choice = random.choices(options, weights, k=1)[0]
        text = text.replace(match.group(0), choice, 1)
    if re.search(r"[{}]", text):
        raise ValueError("Brackets are not matching")
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
