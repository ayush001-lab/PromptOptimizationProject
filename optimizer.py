"""
optimizer.py — version 3

WHAT CHANGED AND WHY

In version 2 the search space was about 240 settings. You could try all of
them with a for-loop, which means Bayesian optimisation was not needed.

The fix is not "more options". It is letting the optimiser learn numbers
that were previously HARD-CODED BY ME.

In version 2 I wrote this and told you to trust it:

    weights = {"NOUN": 3.0, "VERB": 2.0, "ADJ": 1.0, "ADV": 0.5}

Where did 3.0 and 0.5 come from? I made them up. That is a guess sitting
in the middle of your method.

Now they are search parameters. The optimiser finds them from data.
That is what Bayesian optimisation is FOR — tuning continuous values that
are too expensive to try exhaustively.

It also gives you a result worth presenting: the learned weights tell you
which parts of speech actually matter for this task.

Set COMPARE_SAMPLERS = True to run TPE against Random and GP.
"""

import itertools
import json

import numpy as np
import ollama
import optuna
import spacy
from sklearn.metrics import accuracy_score

from dataset import dataset

MODEL = "mistral"
LAMBDA = 0.4
N_TRIALS = 60
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

COMPARE_SAMPLERS = False     # True = also run Random and GP, to prove TPE helps

_cache = {}
_nlp = spacy.load("en_core_web_sm")


# =====================================================================
# MODEL + TOKENS
# =====================================================================

def llama_generate(prompt):
    if prompt in _cache:
        return _cache[prompt]
    r = ollama.chat(model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0})
    _cache[prompt] = r["message"]["content"].strip()
    return _cache[prompt]


try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoder = None


def count_tokens(text):
    return len(_encoder.encode(text)) if _encoder else int(len(text.split()) * 1.3)


# spaCy is slow, so parse each sentence only once and remember the result
_parsed = {}


def parse(text):
    if text not in _parsed:
        _parsed[text] = _nlp(text)
    return _parsed[text]


# =====================================================================
# SCORING WORDS
# =====================================================================

def score_by_pos(text, weights):
    """Score each word using LEARNED weights instead of my guesses.

    `weights` is a dictionary the optimiser fills in, e.g.
        {"VERB": 0.8, "ADJ": 0.2, "ADV": 0.05, ...}
    Nouns are pinned at 1.0 as the reference point, because only the
    RATIOS between weights matter — we keep the top-scoring words, so
    doubling every weight would change nothing.
    """
    out = []
    for t in parse(text):
        if t.is_space:
            continue
        if t.is_stop:
            score = weights["STOP"]
        else:
            score = weights.get(t.pos_, weights["OTHER"])
        out.append((t.text, score))
    return out


def score_by_stopwords(text, weights=None):
    return [(t.text, 0.0 if t.is_stop else 1.0)
            for t in parse(text) if not t.is_space]


_surprisal_cache = {}
_surprisal_models = {}


def score_by_surprisal(text, model_name="distilgpt2"):
    """Score each word by how UNEXPECTED it was to a small model.

    A word the small model could already guess is a word the big model
    can guess too — so sending it is wasted tokens. Words that surprise
    the small model carry the actual information.
    """
    key = (text, model_name)
    if key in _surprisal_cache:
        return _surprisal_cache[key]

    if model_name not in _surprisal_models:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name).eval()
        _surprisal_models[model_name] = (tok, mdl, torch)

    tok, mdl, torch = _surprisal_models[model_name]

    enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
              truncation=True, max_length=512)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        logits = mdl(ids).logits

    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = ids[0, 1:]
    bits = (-logprobs[range(len(targets)), targets] / np.log(2)).tolist()
    per_token = [99.0] + bits          # nothing comes before the first token

    # the model splits words into pieces — average them back into words
    out = []
    for t in parse(text):
        if t.is_space:
            continue
        s0, e0 = t.idx, t.idx + len(t.text)
        vals = [per_token[i] for i, (s, e) in enumerate(offsets)
                if i < len(per_token) and e > s and s < e0 and e > s0]
        out.append((t.text, float(np.mean(vals)) if vals else 0.0))

    _surprisal_cache[key] = out
    return out


# =====================================================================
# BUILDING THE SHORTER SENTENCE
# =====================================================================

def build(text, p):
    """Score the words, then keep the best ones."""
    if p["method"] == "pos":
        scored = score_by_pos(text, p["weights"])
    elif p["method"] == "stopwords":
        scored = score_by_stopwords(text)
    else:
        scored = score_by_surprisal(text, p["scorer_model"])

    doc = parse(text)

    if p["drop_punctuation"]:
        punct = {t.text for t in doc if t.is_punct}
        scored = [(w, s) for w, s in scored if w not in punct]

    if p["protect_entities"]:
        names = set()
        for e in doc.ents:
            names.update(e.text.split())
        scored = [(w, 999.0 if w in names else s) for w, s in scored]

    if p["protect_numbers"]:
        scored = [(w, 999.0 if any(c.isdigit() for c in w) else s)
                  for w, s in scored]

    # how many words survive
    how_many = max(p["min_words"], int(len(scored) * p["ratio"]))
    how_many = min(how_many, len(scored))

    ranked = sorted(range(len(scored)), key=lambda i: -scored[i][1])
    chosen = sorted(ranked[:how_many])          # back into reading order
    return " ".join(scored[i][0] for i in chosen)


def compress_with_llm(text, p):
    return llama_generate(f"""
Rewrite the text to reduce tokens.
Target compression ratio: {p['ratio']}.
Style: {p['style']}.
Template: {p['template']}.
Do NOT change meaning.

Text:
{text}
""")


def compress(text, p):
    return compress_with_llm(text, p) if p["method"] == "llm" else build(text, p)


# =====================================================================
# CLASSIFYING
# =====================================================================

def clean_label(answer):
    low = answer.strip().lower()
    for label in LABELS:
        if label.lower() in low:
            return label
    return "UNKNOWN"


def classify(text):
    return clean_label(llama_generate(f"""
Classify into one of:
World, Sports, Business, Sci/Tech.

Text:
{text}

Answer only label.
"""))


# =====================================================================
# DATA
# =====================================================================

split = int(len(dataset) * 0.7)
train_data, test_data = dataset[:split], dataset[split:]
train_labels = [d["label"] for d in train_data]
test_labels = [d["label"] for d in test_data]

print(f"Train {len(train_data)} / Test {len(test_data)}")
print("Computing baseline (no compression)...")
baseline_acc = accuracy_score(train_labels, [classify(d["text"]) for d in train_data])
print(f"Baseline accuracy: {baseline_acc:.3f}\n")


# =====================================================================
# WHICH METHODS ARE AVAILABLE
# =====================================================================

METHODS = ["llm", "stopwords", "pos"]
try:
    score_by_surprisal("This is a short test sentence.")
    METHODS.append("surprisal")
    print("All four compression methods available.\n")
except Exception as e:
    print(f"!! 'surprisal' unavailable ({type(e).__name__}).")
    print("   pip install torch transformers, then run once with internet.\n")


# =====================================================================
# THE SEARCH SPACE
# =====================================================================

POS_TAGS = ["VERB", "ADJ", "ADV", "NUM", "PROPN", "OTHER", "STOP"]


def suggest(trial):
    p = {"method": trial.suggest_categorical("method", METHODS)}

    # --- settings that apply to every method
    p["ratio"] = trial.suggest_float("ratio", 0.20, 0.95)
    p["min_words"] = trial.suggest_int("min_words", 2, 12)
    p["protect_entities"] = trial.suggest_categorical("protect_entities", [True, False])
    p["protect_numbers"] = trial.suggest_categorical("protect_numbers", [True, False])
    p["drop_punctuation"] = trial.suggest_categorical("drop_punctuation", [True, False])

    # --- settings that ONLY EXIST for certain methods.
    # This is the conditional structure. It is why TPE is the right tool:
    # a Gaussian Process needs every point to have the same coordinates,
    # and here a trial using "pos" has seven numbers that a trial using
    # "llm" simply does not have.
    if p["method"] == "llm":
        p["style"] = trial.suggest_categorical(
            "style", ["concise summary", "entity focused", "remove redundancy"])
        p["template"] = trial.suggest_categorical("template", ["plain", "compact"])

    elif p["method"] == "pos":
        # THE MAIN ADDITION: seven learned weights instead of my guesses.
        # NOUN is pinned at 1.0 as the reference; the rest are relative.
        p["weights"] = {"NOUN": 1.0}
        for tag in POS_TAGS:
            p["weights"][tag] = trial.suggest_float(f"w_{tag.lower()}", 0.0, 1.5)

    elif p["method"] == "surprisal":
        p["scorer_model"] = trial.suggest_categorical(
            "scorer_model", ["distilgpt2", "gpt2"])

    return p


def space_size():
    """How many settings are there, treating each float as 20 steps?
    Print this number in your report."""
    common = 20 * 11 * 2 * 2 * 2          # ratio, min_words, 3 booleans
    llm = 3 * 2
    pos = 20 ** len(POS_TAGS)             # seven continuous weights
    stop = 1
    surp = 2 if "surprisal" in METHODS else 0
    return common * (llm + pos + stop + surp)


print(f"Search space: about {space_size():,.0f} settings")
print("Trying them all is impossible. This is why you need an optimiser.\n")


# =====================================================================
# THE OBJECTIVE
# =====================================================================

def objective(trial):
    p = suggest(trial)

    preds, ratios = [], []
    for item in train_data:
        short = compress(item["text"], p)
        preds.append(classify(short))
        ratios.append(count_tokens(short) / count_tokens(item["text"]))

    acc = accuracy_score(train_labels, preds)
    avg_ratio = float(np.mean(ratios))

    trial.set_user_attr("accuracy", acc)
    trial.set_user_attr("token_ratio", avg_ratio)
    trial.set_user_attr("ratio_error", abs(avg_ratio - p["ratio"]))

    return acc - LAMBDA * avg_ratio


# =====================================================================
# RUN
# =====================================================================

def run(sampler, name):
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))
    done = [t for t in study.trials if t.value is not None]
    print(f"  {name:<8} best {study.best_value:.4f}  ({len(done)} trials)")
    return study


print("Running TPE...")
study = run(optuna.samplers.TPESampler(seed=0, n_startup_trials=15), "TPE")

if COMPARE_SAMPLERS:
    print("\nComparing against other samplers on the same problem...")
    random_study = run(optuna.samplers.RandomSampler(seed=0), "Random")
    try:
        gp_study = run(optuna.samplers.GPSampler(seed=0), "GP")
    except Exception as e:
        print(f"  GP failed: {e}")
        print("  (Optuna's docs say GPSampler has limited support for")
        print("   categorical and conditional spaces — which is your point.)")

    print("\nIf TPE beats Random, the optimiser is doing real work.")
    print("If it does not, say so honestly — that is still a finding.")


best = study.best_params
with open("best_policy.json", "w") as f:
    json.dump(best, f, indent=2)


# =====================================================================
# RESULTS
# =====================================================================

print("\n" + "=" * 66)
print("WHICH COMPRESSION METHOD WON?")
print("=" * 66)
print(f"{'method':<12}{'tried':>7}{'best score':>12}{'accuracy':>10}"
      f"{'tokens':>9}{'ratio gap':>11}")

for method in METHODS:
    ts = [t for t in study.trials
          if t.value is not None and t.params.get("method") == method]
    if not ts:
        print(f"{method:<12}{0:>7}   (none completed)")
        continue
    top = max(ts, key=lambda t: t.value)
    print(f"{method:<12}{len(ts):>7}{top.value:>12.4f}"
          f"{top.user_attrs['accuracy']:>10.3f}"
          f"{top.user_attrs['token_ratio']:>9.1%}"
          f"{np.mean([t.user_attrs['ratio_error'] for t in ts]):>11.3f}")

# The learned weights — this is the interpretable result.
if best.get("method") == "pos":
    print("\n" + "=" * 66)
    print("WEIGHTS THE OPTIMISER LEARNED  (NOUN = 1.0 is the reference)")
    print("=" * 66)
    for tag in POS_TAGS:
        w = best[f"w_{tag.lower()}"]
        bar = "#" * int(w * 30)
        print(f"  {tag:<8}{w:>6.2f}  {bar}")
    print("\nRead this as: which kinds of word survive compression, learned")
    print("from data rather than assumed. A low weight means the optimiser")
    print("found that dropping those words costs little accuracy.")

print(f"\nBaseline accuracy: {baseline_acc:.3f}")
print(f"Best accuracy:     {study.best_trial.user_attrs['accuracy']:.3f}")
print(f"Tokens used:       {study.best_trial.user_attrs['token_ratio']:.1%}")

with open("all_trials.csv", "w") as f:
    f.write("trial,method,ratio,accuracy,token_ratio,ratio_error,score\n")
    for t in study.trials:
        if t.value is None:
            continue
        f.write(f"{t.number},{t.params.get('method')},{t.params.get('ratio'):.4f},"
                f"{t.user_attrs['accuracy']:.4f},{t.user_attrs['token_ratio']:.4f},"
                f"{t.user_attrs['ratio_error']:.4f},{t.value:.4f}\n")
print("\nSaved best_policy.json and all_trials.csv")