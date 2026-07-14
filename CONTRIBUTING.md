# Contributing

Thanks for looking at Sentinel. It's an active, evolving project — happy to
have help.

## Before you start

- Read [README.md](README.md) for the shape of the system and
  [PROJECT_STATUS.md](PROJECT_STATUS.md) for what's real vs. what's an honest
  stub. Don't build on a stub without checking it's actually the right layer.
- For anything non-trivial, open an issue first describing what you want to
  change and why. Saves both of us a wasted PR.
- Small fixes (typos, obvious bugs, a missing test) can just be a PR.

## Working on it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m pytest tests/ -v
```

Everything in this repo runs free — no paid API keys, no special hardware.
If your change needs one of those, say so in the PR and keep it behind a flag
that defaults to the free path.

## PRs

- Keep them focused — one change, one PR.
- Add or update a test for whatever you touched. If it can't be tested (needs
  real hardware, a paid API, etc.), say so in the PR description.
- Match the existing code style: no comments, no docstrings — keep names
  and code clear enough to not need them.
- Run `python -m pytest tests/ -v` before pushing. All green, or explain why not.

## Reporting bugs

Open an issue with what you ran, what you expected, and what happened
instead. A traceback or a failing test is worth a thousand words.
