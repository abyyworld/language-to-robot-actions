# Results

> **Synthetic planners present (noisy_0.5-none, noisy_0.5-symbolic, noisy_0.5-twin).** They read the ground truth and exist to test the pipeline. Their numbers say nothing about language models.

## Runs

| run | condition | episodes | success (95% CI) | API errors | mean rounds | tokens in / out |
|---|---|---:|---|---:|---:|---|
| noisy_0.5-none | noisy:0.5, verifier=none, prompt=basic | 300 | 48% [43%, 54%] | 0 | 1.00 | 0 / 0 |
| noisy_0.5-symbolic | noisy:0.5, verifier=symbolic, prompt=basic, repairs=2 | 300 | 81% [77%, 85%] | 0 | 1.45 | 0 / 0 |
| noisy_0.5-twin | noisy:0.5, verifier=twin, prompt=basic, repairs=2 | 300 | 81% [77%, 85%] | 0 | 1.45 | 0 / 0 |
| oracle-symbolic | oracle, verifier=symbolic, prompt=basic, repairs=2 | 300 | 100% [99%, 100%] | 0 | 1.00 | 0 / 0 |

## By category

| category | noisy_0.5-none | noisy_0.5-symbolic | noisy_0.5-twin | oracle-symbolic |
|---|---|---|---|---|
| simple | 42% [29%, 58%] | 78% [62%, 88%] | 78% [62%, 88%] | 100% [91%, 100%] |
| multi_step | 56% [42%, 69%] | 74% [60%, 84%] | 74% [60%, 84%] | 100% [93%, 100%] |
| precondition | 30% [17%, 48%] | 80% [63%, 90%] | 80% [63%, 90%] | 100% [89%, 100%] |
| spatial | 55% [44%, 65%] | 79% [69%, 86%] | 79% [69%, 86%] | 100% [95%, 100%] |
| counting | 48% [36%, 61%] | 82% [70%, 89%] | 82% [70%, 89%] | 100% [94%, 100%] |
| infeasible | 45% [31%, 60%] | 100% [91%, 100%] | 100% [91%, 100%] | 100% [91%, 100%] |

## Where episodes end

| stage | noisy_0.5-none | noisy_0.5-symbolic | noisy_0.5-twin | oracle-symbolic |
|---|---:|---:|---:|---:|
| success | 145 | 244 | 244 | 300 |
| verifier_rejected | 0 | 9 | 9 | 0 |
| declared_infeasible | 27 | 27 | 27 | 0 |
| execution_invalid | 62 | 0 | 0 | 0 |
| execution_physical | 39 | 0 | 0 | 0 |
| goal_not_met | 20 | 20 | 20 | 0 |
| acted_on_infeasible | 7 | 0 | 0 | 0 |

## What the verifier catches in first drafts

Rule violations in the planner's *first* plan, before any feedback. Repair rate is the share of rejected first drafts that a later round fixed.

| run | first drafts rejected | repaired | top violations |
|---|---:|---:|---|
| noisy_0.5-symbolic | 108/300 | 97/108 | unknown_object (40), off_table (32), hand_empty (23), not_clear (5), unreachable (5) |
| noisy_0.5-twin | 108/300 | 97/108 | unknown_object (40), off_table (32), hand_empty (23), not_clear (5), unreachable (5) |
| oracle-symbolic | 0/300 | - | - |

## Paired comparisons

Each run against **noisy_0.5-none**. Equivalence margin ±5%.

```
noisy_0.5-symbolic vs noisy_0.5-none: 81.3% vs 48.3% (+33.0%, 95% CI [+26.3%, +40.0%], McNemar p=3.16e-30, 99 wins / 0 losses) -> BETTER

group               n    base    cand    diff  95% CI                W/L   p(adj)  verdict
------------------------------------------------------------------------------------------
overall           300   48.3%   81.3%  +33.0%  [+26%, +40%]         99/0 3.16e-30  BETTER
simple             40   42.5%   77.5%  +35.0%  [+20%, +50%]         14/0 0.000244  BETTER
multi_step         50   56.0%   74.0%  +18.0%  [+8%, +30%]           9/0  0.00391  BETTER
precondition       30   30.0%   80.0%  +50.0%  [+30%, +67%]         15/0 0.000183  BETTER
spatial            80   55.0%   78.8%  +23.8%  [+15%, +34%]         19/0 1.53e-05  BETTER
counting           60   48.3%   81.7%  +33.3%  [+18%, +50%]         20/0 9.54e-06  BETTER
infeasible         40   45.0%  100.0%  +55.0%  [+40%, +70%]         22/0 2.86e-06  BETTER
```

```
noisy_0.5-twin vs noisy_0.5-none: 81.3% vs 48.3% (+33.0%, 95% CI [+26.3%, +40.0%], McNemar p=3.16e-30, 99 wins / 0 losses) -> BETTER

group               n    base    cand    diff  95% CI                W/L   p(adj)  verdict
------------------------------------------------------------------------------------------
overall           300   48.3%   81.3%  +33.0%  [+26%, +40%]         99/0 3.16e-30  BETTER
simple             40   42.5%   77.5%  +35.0%  [+20%, +50%]         14/0 0.000244  BETTER
multi_step         50   56.0%   74.0%  +18.0%  [+8%, +30%]           9/0  0.00391  BETTER
precondition       30   30.0%   80.0%  +50.0%  [+30%, +67%]         15/0 0.000183  BETTER
spatial            80   55.0%   78.8%  +23.8%  [+15%, +34%]         19/0 1.53e-05  BETTER
counting           60   48.3%   81.7%  +33.3%  [+18%, +50%]         20/0 9.54e-06  BETTER
infeasible         40   45.0%  100.0%  +55.0%  [+40%, +70%]         22/0 2.86e-06  BETTER
```

```
oracle-symbolic vs noisy_0.5-none: 100.0% vs 48.3% (+51.7%, 95% CI [+46.0%, +57.3%], McNemar p=4.38e-47, 155 wins / 0 losses) -> BETTER

group               n    base    cand    diff  95% CI                W/L   p(adj)  verdict
------------------------------------------------------------------------------------------
overall           300   48.3%  100.0%  +51.7%  [+46%, +57%]        155/0 4.38e-47  BETTER
simple             40   42.5%  100.0%  +57.5%  [+42%, +72%]         23/0 9.54e-07  BETTER
multi_step         50   56.0%  100.0%  +44.0%  [+30%, +58%]         22/0 1.43e-06  BETTER
precondition       30   30.0%  100.0%  +70.0%  [+53%, +87%]         21/0 1.43e-06  BETTER
spatial            80   55.0%  100.0%  +45.0%  [+34%, +56%]         36/0 1.75e-10  BETTER
counting           60   48.3%  100.0%  +51.7%  [+38%, +63%]         31/0 4.66e-09  BETTER
infeasible         40   45.0%  100.0%  +55.0%  [+40%, +70%]         22/0 1.43e-06  BETTER
```


![success by category](report.png)

