"""Score Akira's offline knowledge on common-knowledge questions.

    python3 evals/knowledge.py

Each answer counts when it contains one of the expected words. "No answer" is better than a wrong
one (Akira then asks the internet), so wrong answers are reported separately.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira.knowledge import Knowledge  # noqa: E402

QUESTIONS = [
    ("What is the capital of France?", ["Paris"]), ("Who was Albert Einstein?", ["physic", "scientist"]),
    ("What is photosynthesis?", ["plants", "sunlight"]), ("How tall is Mount Everest?", ["8,8", "8848", "29,0"]),
    ("When did World War II end?", ["1945"]), ("What is the largest planet?", ["Jupiter"]),
    ("Who wrote Romeo and Juliet?", ["Shakespeare"]), ("What is the boiling point of water?", ["100"]),
    ("What language is spoken in Brazil?", ["Portuguese"]), ("What is DNA?", ["genetic", "molecule"]),
    ("Who painted the Mona Lisa?", ["Leonardo"]), ("What is the speed of light?", ["299", "300,000", "186,"]),
    ("What is the capital of Japan?", ["Tokyo"]), ("Who invented the telephone?", ["Bell"]),
    ("What is a volcano?", ["lava", "magma"]), ("How many legs does a spider have?", ["eight", "8"]),
    ("Who was the first president of the United States?", ["Washington"]),
    ("What is the chemical symbol for gold?", ["Au"]), ("What is gravity?", ["force", "pull", "attract"]),
    ("Where is the Eiffel Tower?", ["Paris"]), ("What is the capital of Italy?", ["Rome"]),
    ("Who discovered penicillin?", ["Fleming"]), ("What is the largest ocean?", ["Pacific"]),
    ("How many continents are there?", ["seven", "7"]), ("What is the freezing point of water?", ["0", "32"]),
    ("Who was Isaac Newton?", ["physic", "mathematic", "scientist"]), ("What is the smallest planet?", ["Mercury"]),
    ("What is the capital of Australia?", ["Canberra"]), ("Who wrote Hamlet?", ["Shakespeare"]),
    ("What is the longest river in the world?", ["Nile", "Amazon"]),
]

k = Knowledge(os.path.join(os.path.dirname(__file__), "..", "models", "knowledge.db"))
right = wrong = none = 0
for q, expected in QUESTIONS:
    a, doc = k.answer(q)
    if a is None:
        none += 1
        mark = "  -"
    elif any(e.lower() in a.lower() for e in expected):
        right += 1
        mark = " ok"
    else:
        wrong += 1
        mark = "BAD"
    print(f"{mark} {q:52} {(a or '')[:90]}")
print(f"\n{right}/{len(QUESTIONS)} right, {wrong} wrong, {none} no answer")
