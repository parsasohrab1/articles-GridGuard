# TRL assessment – GridGuard

**Claimed level: TRL 4** (laboratory validation on a public **simulated** dataset). Not TRL 5: no measured grid data, no relevant operating environment.

Data: UCI 471, n = 10000 simulated parameter sets, 64% unstable. Shift split: train/calibration mean reaction time <= 5.62 (unstable share 48%); test >= 6.43 (unstable share 92%). Class-wise conformal, alpha = 0.05, mean of 30 repeats.

| Model | Setting | Coverage unstable | Coverage stable | Referral rate | Accuracy on automatic | Unstable called stable (conformal, automatic) | Unstable called stable (argmax) |
|---|---|---|---|---|---|---|---|
| logistic | iid | 0.951 | 0.949 | 0.413 | 0.916 | 0.049 | 0.122 |
| logistic | shift | 1.000 | 0.766 | 0.138 | 0.979 | 0.000 | 0.000 |
| grad_boosting | iid | 0.949 | 0.952 | 0.095 | 0.945 | 0.051 | 0.043 |
| grad_boosting | shift | 0.926 | 0.975 | 0.095 | 0.923 | 0.074 | 0.112 |

## Findings and caveats
- Under random splits the class-wise guarantee holds (about 0.95 per class). The price is high for the linear model (41 % referrals) and moderate for gradient boosting (9.5 %).
- Under the reaction-time shift the guarantee for the dangerous class is lost for gradient boosting (0.926 instead of 0.95; 7.4 % of unstable cases decided automatically as stable, against 11.2 % for plain argmax). The logistic model keeps unstable coverage (1.0) only because its sets become very large on the stable class (stable coverage 0.766 – the guarantee fails for the other class).
- The shift also changes class prevalence (48 % to 92 % unstable), so the experiment mixes covariate and prior shift.
- The label is a deterministic function of the parameters (sign of the largest eigenvalue real part), so a classifier is a surrogate for a known equation; its value is speed, not new information.
- Everything is simulated: results say nothing about measured grids.
