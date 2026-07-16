"""Bot swarm — population-scale pattern search with honest evaluation.

Thousands of bots, each a tiny hypothesis (2-4 market features, 1-3 sampled
rules) wrapped in a behavioral genome (sizing, stops, patience, sessions...),
plus a random placebo group. Judged strictly out-of-sample.

See bot-swarm-discussion.md at the repo root for the full design rationale.
"""
