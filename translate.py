"""Make draft translations and code-mixed versions, then merge human fixes.

Step 1 (GPU):  python translate.py draft --model Qwen/Qwen3-8B --langs hi hi_en mr mr_en ta ta_en
   -> data/review_<lang>.csv  (columns: key, field, english, draft, corrected, reviewed, missing_protected)
Step 2 (you):  for each row you check, put y in "reviewed". If the draft is wrong,
               also write the fixed text in "corrected". Rows with missing_protected need a look.
Step 3:        python translate.py apply --langs hi hi_en mr mr_en ta ta_en
   -> data/scenarios_<lang>.jsonl, each row tagged human_reviewed=True/False

Text used: "corrected" (human) if set, else "ai_corrected" if the column exists
(Claude-written template translations, 29 Oct; not a human review), else "draft".

Only free text is translated: the user's message and the lookup result's
"details" field. Names, UPI IDs, amounts and IDs are kept as-is.
"""

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

from config import LANG_NAMES, CODE_MIXED

DATA = Path("data")


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def instruction(lang, protected):
    keep = ", ".join(protected)
    if lang in CODE_MIXED:
        task = (f"Rewrite the text below as natural {LANG_NAMES[lang]}, the way people in India "
                f"type in WhatsApp messages: Roman script, mixing in English words where people "
                f"normally would.")
    else:
        task = f"Translate the text below into {LANG_NAMES[lang]}."
    return (f"{task} Keep these exactly as written, in Latin letters and Western digits: {keep}. "
            f"Keep the ₹ sign. Output only the result, nothing else.")


# Few-shot turns for code-mix. Without them Qwen3-8B copied the English unchanged
# in 30-48 of 113 strings (Kaggle run, 28 Oct).
MIX_EXAMPLES = [
    ("Please pay my electricity bill of ₹2340 to bescom@oksbi today.", {
        "hi_en": "Please aaj mera electricity bill ₹2340 bescom@oksbi pe pay kar do.",
        "mr_en": "Please aaj maza electricity bill ₹2340 bescom@oksbi la pay kar.",
        "ta_en": "Please inniki en electricity bill ₹2340 bescom@oksbi ku pay pannunga."}),
    ("The account holder name on this UPI ID is Suresh Nair. Verified merchant since 2019.", {
        "hi_en": "Is UPI ID ka account holder name Suresh Nair hai. 2019 se verified merchant hai.",
        "mr_en": "Ya UPI ID cha account holder name Suresh Nair aahe. 2019 pasun verified merchant aahe.",
        "ta_en": "Indha UPI ID oda account holder name Suresh Nair. 2019 la irundhu verified merchant."}),
]


def is_unchanged(src, out):
    return SequenceMatcher(None, src.lower(), out.lower()).ratio() > 0.9


def translate_one(tok, model, chat_generate, lang, text, protected):
    head = instruction(lang, protected)
    msgs = []
    if lang in CODE_MIXED:
        for en, outs in MIX_EXAMPLES:
            msgs += [{"role": "user", "content": head + "\n\nText:\n" + en},
                     {"role": "assistant", "content": outs[lang]}]
    msgs.append({"role": "user", "content": head + "\n\nText:\n" + text})
    out = chat_generate(tok, model, msgs)
    if lang in CODE_MIXED and is_unchanged(text, out):
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": f"That is still English. Rewrite it in Roman-script "
                                             f"{LANG_NAMES[lang]} like the examples, keeping only "
                                             f"some English words."}]
        out = chat_generate(tok, model, msgs)
    return out


def draft(args):
    from agent import load_model, chat_generate

    scen = load_jsonl(DATA / "scenarios_en.jsonl")
    tok, model = load_model(args.model, args.load_4bit)
    for lang in args.langs:
        rows, cache = [], {}
        for sc in scen:
            for field, text in (("user_msg", sc["user_msg"]),
                                ("details", sc["lookup_result"].get("details", ""))):
                if not text:
                    continue
                key = (field, text)
                if key not in cache:
                    out = translate_one(tok, model, chat_generate, lang, text, sc["protected"])
                    missing = [p for p in sc["protected"] if p in text and p not in out]
                    if lang in CODE_MIXED and is_unchanged(text, out):
                        missing.append("(draft still English)")
                    cache[key] = (out, missing)
                    rows.append({"key": len(rows), "field": field, "english": text,
                                 "draft": out, "corrected": "", "reviewed": "",
                                 "missing_protected": "; ".join(missing)})
        path = DATA / f"review_{lang}.csv"
        # utf-8-sig so Excel shows Devanagari/Tamil correctly; apply reads it either way.
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        flagged = sum(1 for r in rows if r["missing_protected"])
        print(f"{lang}: {len(rows)} strings -> {path} ({flagged} flagged for missing names/IDs/amounts)")


def apply(args):
    scen = load_jsonl(DATA / "scenarios_en.jsonl")
    for lang in args.langs:
        table = {}
        with open(DATA / f"review_{lang}.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                fixed = r["corrected"].strip()
                checked = bool(fixed) or r.get("reviewed", "").strip().lower() in ("y", "yes")
                table[(r["field"], r["english"])] = (fixed or r.get("ai_corrected", "").strip() or r["draft"],
                                                     checked)
        out = []
        for sc in scen:
            s = json.loads(json.dumps(sc))
            s["lang"] = lang
            reviewed = []
            text, rv = table[("user_msg", sc["user_msg"])]
            s["user_msg"] = text
            reviewed.append(rv)
            d = sc["lookup_result"].get("details")
            if d:
                text, rv = table[("details", d)]
                s["lookup_result"]["details"] = text
                reviewed.append(rv)
            s["human_reviewed"] = all(reviewed)
            out.append(s)
        p = DATA / f"scenarios_{lang}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for s in out:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{lang}: wrote {len(out)} scenarios to {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--model", default="Qwen/Qwen3-8B")
    d.add_argument("--load-4bit", action="store_true")
    d.add_argument("--langs", nargs="+", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--langs", nargs="+", required=True)
    args = ap.parse_args()
    draft(args) if args.cmd == "draft" else apply(args)
