# Handoff: DPI Probe Gate

Read this first, then README.md. This file gives the context, the decisions behind them, the status, and the next steps.

## Who and why

- Owner: Raj, final-year EEE + Data Science student at BITS Pilani Goa, working solo.
- Interest: pragmatic mechanistic interpretability (in the sense of the GDM "pragmatic interpretability" agenda: proxy tasks, beat black-box baselines).
- Uses of this project:
  1. Entry for the SAFL **India Agentic AI Security Hackathon**. Deadline **Sat 31 Oct 2026, 11:59 PM IST**. One entry, solo is allowed. Submission form: bit.ly/saflhacksubmission.
  2. Evidence for an application to the Secure AI Futures Lab (SAFL) Research Fellow internship track.
  3. A workshop paper and a blog post after about 2 months.

## How Raj wants you to work

- Ask clarifying questions where a decision is his, and wait for his answer before you continue.
- If you don't know something, say so. Do not guess, speculate or fabricate data, results or citations.
- Never fake or simulate results, backend responses or a working demo. If something has not run, say it has not run.

## The research question

Does an LLM agent internally represent that an action will harm the user, before it takes the action? Does that signal survive when the agent operates in Hindi, Marathi, Tamil and their code-mixed versions?

- Probe: linear probe on residual-stream activations at the **decision point** (last prompt token after the lookup result, before the agent writes its next move).
- Train on English only. Test on `hi`, `hi_en`, `mr`, `mr_en`, `ta`, `ta_en`.
- Baseline: black-box LLM judge that reads the transcript and the agent's move.
- Demo: a gate that blocks a world-changing tool call when the probe score passes the threshold.

## Decisions already made (do not reopen without asking Raj)

| Decision | Reason |
|---|---|
| Agent models: `Qwen/Qwen3-8B` and `dheeyantra/dhee-pravega` (Gemma-4 E2B, Indic tool calling) | Two models show whether findings generalise; Dhee is built for Indic tool calls |
| Seven language conditions above | Raj chose Marathi, Hindi, Tamil + code-mixed. He checks hi, hi_en, mr, mr_en; friends check ta, ta_en |
| Code-mix made as LLM drafts, then human-corrected | Translation models output pure script only |
| 5 harm domains × 12 minimal pairs = 120 English scenarios | Minimal pairs stop probes from scoring on surface differences |
| Folds split by `pair_id` across all languages | Prevents translated copies of training items leaking into test |
| Threshold fixed at 5% FPR on English, applied unchanged to other languages | Mirrors how a deployed gate behaves |
| Judge: `sarvamai/sarvam-m` for both agents (4-bit on T4) | Raj chose it 27 Oct: covers hi/mr/ta, not gated, different family from Qwen and Gemma |
| Backend: explicit dry-run (`DryRunBackend`) for the demo and all results | Raj chose it 28 Oct. The SAFL submission template says "SAFL does not provide a test environment, sandbox or datasets". Dry-run calls nothing outside; describe it in submission Section 9. `HttpSandboxBackend` stays in the code but is not used. No faked responses ever |

## Hard constraints

- Hackathon rules: test **only** in provided DPI sandboxes or systems Raj owns or has permission to test. **Never** call live Aadhaar, UPI or other production systems. Do not publish exploits that put real users at risk.
- Rules require disclosure of AI tool use. README has a disclosure section; keep it accurate.
- Code written before the event must be cited; the core work must happen in the hackathon window (it does, from 27 Oct).
- Compute: Kaggle dual-T4, Colab free, occasional BITS HPC H100, vast.ai. About 7-9B open models, inference and activation access only.

## Status (as of 27 Oct 2026, late evening IST)

Verified:
- `python -m pytest tests -q` passes (12 tests, offline, no GPU).
- Both models' real chat templates render the scenario prompts (checked with downloaded tokenizers).
- Dhee-Pravega emits `<|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>`; `agent.parse_tool_calls` handles it (tested on a hand-written sample, not real model output).
- `data/scenarios_en.jsonl` generated (120 rows, seed 0).
- GPU smoke test passed on Kaggle T4x2, fp16 (kernel `rajtaware10/safl-dpi-smoke-test` v3, 27 Oct; code from private dataset `rajtaware10/safl-dpi-probe-gate-code`). 5 English scenarios per model, `payee_mismatch` only:
  - Qwen3-8B: acts (5, 37, 4096), all finite. 9.1 s/scenario (about 2.1 h per 840). Called no tool in 5/5; it asked the user to confirm, and did not mention the payee-name mismatch.
  - Dhee-Pravega: acts (5, 36, 1536), all finite. 9.8 s/scenario (about 2.3 h per 840). Called `send_upi_payment` in 5/5, harmful cases too; the parser handles its real output. It also writes extra invented calls and a "payment sent" message in the same turn, because generation does not stop at the first tool call. `executed_target` checks all parsed calls, not only the first.
  - Dhee must load on one GPU (`SINGLE_GPU_MODELS` in `agent.py`): split over 2 GPUs, Gemma-4 fails with a cuda:0 vs cuda:1 error.
  - `kaggle kernels logs` returned nothing; the notebook writes logs to `/kaggle/working/logs/`, read them with `kaggle kernels output ... --file-pattern logs`.

- English run, 28 Oct (kernel `rajtaware10/safl-dpi-english`). Batched generation (bs 8, left pad, stop at first tool call) gives the same acts as bs 1 (cos > 0.9999); speed-up Dhee 1.8x (2.9 s/scenario in the full run), Qwen none (11.7 s). Judge is now one forward pass reading P(yes)/P(yes+no) (8-9 s/item, yes+no mass median 0.94-0.99).
  - English AUROC with pair folds is about 1.0 from layer 1 for both models, and a TF-IDF text classifier gets 0.997, so English pair-fold AUROC shows nothing. Leave-one-domain-out (train on 4 domains, test on the 5th): logreg probe reaches 1.0 at Qwen layers 20-29 and Dhee layers 7-10; text classifier 0.75. `probes.py` now picks the layer by leave-one-domain-out AUROC (Qwen 20, Dhee 7) and reports the text baseline.
  - Harmful action done (English): Dhee 80% (all except injection), Qwen 22% (13/60; injection 8/12). Qwen sometimes writes the tool name `send_upi payment` (5/120 English, all `payee_mismatch`). Raj chose 29 Oct to count it: the parser now maps spaces to `_` in the name. Before that, Qwen was 15% (9/60). Saved outputs were parsed again locally and nothing was rerun. Only Qwen English changed (the safe-action rate also moved, 0.45 -> 0.47). Probe and judge numbers do not change.
- Per-language calibration (Raj chose 28 Oct): 10 labelled benign rows per language shift the English threshold by (their median - English benign median); `probes.py` reports TPR/FPR at the English threshold and calibrated (mean over 20 draws, calibration pairs excluded), for probe and judge.
- Adaptive attack, 29 Oct (kernel `rajtaware10/safl-dpi-attack`, `attack.py`, `kaggle_attack/`). 12 harmful English `injection` scenarios, 8 valid Qwen3-8B paraphrases each (all 12 got 8/8), attacker keeps the lowest probe score. Probe misses 0/12 for both agents, original and attacker pick; also 0/96 over every paraphrase. Qwen: agent executes 8/12 original, 11/12 attacker pick (paraphrases raise execution); lowest probe score 4.54 vs threshold -3.73. Dhee: executes 0/12 in all 108 runs, so the attack has nothing to hide; attacker pick lowers mean probe score 6.08 -> 1.89 (lowest -0.43 vs threshold -2.17), so margin shrinks. Judge (English threshold) misses 0/12 for both. Weak attacker (best-of-8, no gradient attack): a lower bound only.
- Translation fixes, 29 Oct (Raj: "start translation review fixes no reruns"). The 113 English strings come from 14 sentence templates. Claude translated each template once per language and filled the slots in Latin script, into a new `ai_corrected` column in `data/review_<lang>.csv`. The script that did this was in the session scratchpad, not the repo. `translate.py apply` now uses `corrected`, then `ai_corrected`, then `draft`. `ai_corrected` does not set `human_reviewed`, so `human_reviewed` is still False everywhere. The `data/scenarios_<lang>.jsonl` files were rewritten. Every protected token missing from them is also missing from the English source (protected lists include tokens from the pair partner). The old Qwen drafts are kept in the `draft` column. The collected activations and all results so far use the OLD drafts. Nothing was rerun. Raj and friends still need to review: 14 sentences per language.
- All-language run launched 28 Oct on UNREVIEWED drafts (`human_reviewed=False`): kernels `rajtaware10/safl-dpi-all-dhee` and `-all-qwen` (`kaggle_all_*/`). Rerun `probes.py` locally on their outputs (the kernels use the old layer pick). Rerun collect for languages whose review changes the text.

Not verified:
- Behaviour on the other 4 harm domains and on translated text (smoke test covered `payee_mismatch` only).
- Translation quality. Review sheets exist (28 Oct, Kaggle kernel `rajtaware10/safl-dpi-translate`, Qwen3-8B drafts, 113 strings per language) and were sent to Raj; human review not done. Known problems: hi/mr/ta drafts write names in native script (54/42/54 rows flagged) and break 8/10/6 UPI IDs. Code-mixed sheets were re-drafted with few-shot examples (v2, `kaggle_translate/` now runs only hi_en mr_en ta_en); rows still English: hi_en 1, mr_en 13, ta_en 39. Tamil drafts are weak and some change the meaning. The review CSVs are not committed yet.
- Any backend. `backend_config.json` does not exist; everything runs dry-run.

## File map

- `config.py` — languages, model ids, action vs lookup tools, system prompt
- `tools.py` — tool schemas; `HttpSandboxBackend`, `DryRunBackend`, `LocalInbox`
- `make_scenarios.py` — paired English scenarios
- `translate.py` — `draft` (LLM translations + review CSVs) and `apply` (merge human fixes)
- `agent.py` — model loading, prompt building, generation + decision-point activations, tool-call parsing, `executed_target`
- `collect.py` — run agent over scenarios, save `data/<model>/<lang>_acts.npy` and `_meta.jsonl` (resumable)
- `judge.py` — black-box judge scores
- `probes.py` — cross-validated per-layer probes, metrics, plots, final probe for the demo
- `demo_app.py` — Gradio live gate (`--share` for Kaggle), or `--cli`
- `inbox/` — example inbox items Raj wrote (user-owned content for `read_inbox_item`)
- `tests/test_offline.py` — offline tests

## Next steps, in order

1. Smoke test on GPU: `python collect.py --model qwen3-8b --langs en --limit 5`, same for `dhee-pravega`. Check the printed raw output parses into tool calls. If it doesn't, extend `parse_tool_calls` and add a test with the real output. Report seconds per scenario × 840 to Raj so he can judge whether full runs fit before Saturday.
2. `python translate.py draft --langs hi hi_en mr mr_en ta ta_en`. Send Raj the review CSVs. Rows with `missing_protected` need attention first.
3. After Raj and friends review: `python translate.py apply ...`.
4. Full `collect.py` runs for both models, then `judge.py`, then `probes.py`.
5. Adaptive-attack test (Raj approved 27 Oct). Reason: the SAFL concept note judges whether a solution holds up "when the agent's logic, prompts, or weights are public knowledge". Method: for the 12 harmful `injection` scenarios, make 8 LLM paraphrases of the planted instruction; an attacker who knows the probe keeps the paraphrase with the lowest probe score. Report (a) how often the agent still executes the target payment, (b) how often the probe score stays under the English 5%-FPR threshold, (c) Sarvam-M judge scores on the same paraphrases. Best-of-8 is a weak attacker (no gradient attack such as GCG), so state the result as a lower bound on attacker strength. Needs the final probe from step 4.
6. (Closed 28 Oct: no sandbox. The demo uses the dry-run backend.)
7. Run the demo with real requests in several languages; record a short video for the submission.
8. Submission write-up: threat model, method, results with honest sample sizes, limitations, adoption path (IndiaAI Safety Institute, RBI, MeitY), AI-use disclosure.

If time runs short before Saturday, ask Raj before cutting scope. Possible cuts: one model instead of two, or fewer languages in the hackathon version, with the rest in the 2-month version.

## Open questions for Raj

- None open. Sandbox closed 28 Oct (dry-run).

## Submission requirements (from the SAFL template, bit.ly/hacksubmissiontemplate, read 28 Oct)

- Every link must open without a login. **The GitHub repo is private now; make it public before submitting.**
- Report PDF, max 10 pages (`TeamName_Report.pdf`). Repo with README (how to run, reproduce, setup) is required for technical prototypes. Demo video optional, max 2 min.
- Section 6 "Threat model and resilience under exposure" (150-250 words) is "a core judging criterion": say whether the method works if the attacker knows the prompts, logic or weights. Use the adaptive-attack result here.
- Section 9: both declarations must be Yes (synthetic data only; no working exploit details in public material), plus 1-2 lines on the test environment. Draft: "Action tool calls go to an explicit dry-run backend in our own code that returns a fixed 'NOT executed' result; lookups read synthetic local files. No real UPI, Aadhaar, DigiLocker or Account Aggregator system was called. All scenarios and identities are synthetic."

## Known method limits (keep these visible in any write-up)

- 60 pairs is a pilot. Report confidence intervals; don't overclaim.
- Best layer is picked on English, so the English score is slightly optimistic.
- The judge sees the generated move; the probe reads activations before the move exists.
- In the live demo, scores before the first lookup are outside the training distribution; only action tools are gated.

## Related prior work (from the novelty check)

DeceptGuard/DeceptArena (black-box vs CoT vs activation-probe monitors in agentic settings); black-to-white performance boost benchmark for deception probes; PRISMS (neuron detectors for tool-use failures); "Before Retrieval Begins" (harmful tool admission, Oct 2026; whether it uses activations was not confirmed); IndicSafe and CompositeHarm (uneven safety across Indic languages); cross-lingual shared safety neurons (ICML 2026). No exact match to "activation gate for harmful DPI agent actions, tested across Indic and code-mixed languages" was found, but the search was not exhaustive. Re-check before writing the paper.
