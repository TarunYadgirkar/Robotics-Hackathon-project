"""Content-word coverage matcher.

Deliberately NOT string-similarity thresholding on the whole query
(that is the known-broken approach: token_set_ratio against a real
bottle-related task name scores well above a naive 0.55 threshold, so
"bottle flip" would wrongly match a cleaning task and BEAT 3 would
invert).

Instead: tokenize the query into content words, and for each task
independently ask "is every content word something this task's name
or aliases actually contains (allowing small spelling fuzz)?". A
word like "flip" never fuzzy-matches any real task vocabulary word,
so it drags coverage below threshold regardless of how similar the
sentence looks as a whole. This is emergent from the real metadata,
not a special case for "bottle flip".
"""
import difflib
import re

STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "and", "or",
    "do", "does", "did", "please", "can", "could", "would", "should", "will",
    "you", "i", "me", "my", "we", "our", "it", "its", "is", "are", "be", "been",
    "this", "that", "these", "those", "like", "want", "wanna", "go", "now",
    "just", "up", "down", "some", "any",
}

FUZZY_MATCH_THRESHOLD = 0.85
COVERAGE_MATCH_THRESHOLD = 0.8


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def content_words(text):
    return [w for w in tokenize(text) if w not in STOPWORDS]


def task_vocab(task):
    words = set(tokenize(task["display_name"]))
    words |= set(tokenize(task["canonical_task_id"].replace("-", " ")))
    for alias in task.get("aliases") or []:
        words |= set(tokenize(alias))
    return words


def word_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def best_vocab_match(word, vocab):
    best_word, best_score = None, 0.0
    for v in vocab:
        score = word_similarity(word, v)
        if score > best_score:
            best_word, best_score = v, score
    return best_word, best_score


def coverage_for_task(query_words, task):
    vocab = task_vocab(task)
    covered, uncovered = [], []
    for word in query_words:
        match_word, score = best_vocab_match(word, vocab)
        entry = {"word": word, "closest": match_word, "score": round(score, 3)}
        if score >= FUZZY_MATCH_THRESHOLD:
            covered.append(entry)
        else:
            uncovered.append(entry)
    coverage = (len(covered) / len(query_words)) if query_words else 0.0
    return coverage, covered, uncovered


def rank_tasks(query_words, tasks):
    ranked = []
    for task in tasks:
        coverage, covered, uncovered = coverage_for_task(query_words, task)
        ranked.append({
            "task_id": task["canonical_task_id"],
            "display_name": task["display_name"],
            "coverage": coverage,
            "covered": covered,
            "uncovered": uncovered,
        })
    ranked.sort(key=lambda r: (-r["coverage"], r["task_id"]))
    return ranked
