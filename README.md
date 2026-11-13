# Polymarket "Next CEO of NVIDIA?" 4-hour-ahead quantile forecaster

Entry for make.fun's `polymarket-nvidia-ceo-forecast-01`.

`main.py` reads one case (`/input/<id>.in`: every market of the event with its Yes price for every
minute of the 72 hours up to a checkpoint) and writes `/output/<id>.out`: for each market and each of
the next 240 minutes, the 5/25/50/75/95 quantiles of that minute's price.

## Model

A price is a bounded probability, so the state space is its logit. For each market the program takes
its own 72 hours of minutes, keeps the minutes where the market sat near where it sits now (a move
measured from a different regime is not evidence about this one), weights the recent past more heavily
than three days ago, and reads the empirical quantiles of its logit moves off that conditioned sample
at a spread of horizons, interpolating between them. The quantiles are mapped back through the sigmoid,
pulled a little way toward a normal with the same interquartile range (5% of a few hundred samples is a
noisy number and the tails are where a wrong band costs most), and scaled by constants searched against
the practice cases and held-out cuts of the same history.

So a market that mostly sits still gets a band that mostly sits still — an atom at the current price is
a feature, not something to smooth away — while a market that jumps gets a fat band at the horizon
where its jumps happen. This beats the house baseline's fixed spread on the practice cases.

Standard library only, offline, deterministic, ~0.15 s for a 24-market case against an 8-second cap.
