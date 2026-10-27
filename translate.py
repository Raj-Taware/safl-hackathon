"""Make draft translations and code-mixed versions, then merge human fixes.

Step 1 (GPU):  python translate.py draft --model Qwen/Qwen3-8B --langs hi hi_en mr mr_en ta ta_en
   -> data/review_<lang>.csv  (columns: key, field, english, draft, corrected, reviewed, missing_protected)
Step 2 (you):  for each row you check, put y in "reviewed". If the draft is wrong,
               also write the fixed text in "corrected". Rows with missing_protected need a look.
Step 3:        python translate.py apply --langs hi hi_en mr mr_en ta ta_en
   -> data/scenarios_<lang>.jsonl, each row tagged human_reviewed=True/False

Only free text is translated: the user's message and the lookup result's
"details" field. Names, UPI IDs, amounts and IDs are kept as-is.
"""

import argparse
import csv
import json
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
                    msgs = [{"role": "user",
                             "content": instruction(lang, sc["protected"]) + "\n\nText:\n" + text}]
                    out = chat_generate(tok, model, msgs)
                    missing = [p for p in sc["protected"] if p in text and p not in out]
                    cache[key] = (out, missing)
                    rows.append({"key": len(rows), "field": field, "english": text,
                                 "draft": out, "corrected": "", "reviewed": "",
                                 "missing_protected": "; ".join(missing)})
        path = DATA / f"review_{lang}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        flagged = sum(1 for r in rows if r["missing_protected"])
        print(f"{lang}: {len(rows)} strings -> {path} ({flagged} flagged for missing names/IDs/amounts)")


def apply(args):
    scen = load_jsonl(DATA / "scenarios_en.jsonl")
    for lang in args.langs:
        table = {}
        with open(DATA / f"review_{lang}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                fixed = r["corrected"].strip()
                checked = bool(fixed) or r.get("reviewed", "").strip().lower() in ("y", "yes")
                table[(r["field"], r["english"])] = (fixed or r["draft"], checked)
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
