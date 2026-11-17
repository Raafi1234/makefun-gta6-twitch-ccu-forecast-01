# GTA VI Twitch CCU forecast — make.fun entry

Forecasts Grand Theft Auto VI's launch-day Twitch viewers for hours 2–24 of
2026-11-19 (UTC) as five quantiles per hour, given only hour 1's print.

`main.py` (Python, standard library only) builds an analogue forecast at run
time from the case's own comparables: for each past launch it forms the
release-day log ratios to that launch's hour 1, then takes weighted empirical
quantiles per hour. Weights favour comparables close in log hour-1 size (a
proxy for unlock shape: rolling-midnight console launches start large, global
unlocks start tiny) and in platform overlap. Quantile curves are smoothed over
a 3-hour window and their spread around the median is scaled by 0.9 — both
calibrated by leave-one-out cross-validation over the comparables (0.404 vs
the persistence baseline's 0.930; 0.262 vs 0.351 on launches with hour 1 above
10,000 viewers).

Run locally:

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
