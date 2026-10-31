# EX.Y: <question as a short title>

> Copy this file to `04-experiments/EX.Y-short-slug.md` (X = roadmap phase, Y = experiment number, matching the roadmap's `EX.Y` IDs when one exists).

- **Roadmap phase**: [Phase X](../00-plan/roadmap.md#phase-x-...) · Hypothesis ladder rung: X
- **Status**: planned | running | done | abandoned
- **Started / finished**: YYYY-MM-DD / YYYY-MM-DD

## Hypothesis

- **H**: <falsifiable statement>
- **Null**: <what you would observe if H is false>

## Why it moves the research forward

<What decision this result changes (model, data, metric, next phase). State what you'd do if H holds and what you'd do if it fails.>

## Knowledge tested

- *Domain*: <biology / medicine concepts>
- *Technical*: <ML / stats concepts>

## Design

- **Data**: <dataset, version, link to [datasets](../03-data/datasets.md)>
- **Splits**: <train/val/test, OOD definition>
- **Models and baselines**: <including the simple baseline: mean, linear, PCA>
- **Code / notebook**: <path or repo + commit>

## Metric and success criterion

<Metric(s), statistical test, and the threshold decided *before* running.>

## Results

<Numbers, figures, seeds, compute. Note anything that deviated from the design.>

## Conclusion

- **Verdict**: H supported | H rejected | inconclusive
- **What I learned**:
- **Progress signal met?**: <yes/no, with a reason>
- **Next step**: <the next experiment or roadmap change, also logged in [research-log](../research-log.md)>
