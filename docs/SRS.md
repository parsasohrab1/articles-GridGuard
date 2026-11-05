# Software Requirements Specification – GridGuard

- SR-1 `data.load_grid`: 11 features (4 reaction times, 3 powers, 4 price elasticities; p1 dropped because it equals the negative sum of the others); target unstable = 1.
- SR-2 `conformal.classwise_thresholds`: split-conformal threshold per true class on 1 - p(class); `sets`: label sets; `decide`: automatic label if the set is a singleton, otherwise -1 (refer).
- SR-3 `research/run_study.py`: logistic regression and gradient boosting; alpha 0.05; 30 repeats of (a) random split 40/30/30 and (b) a covariate-shift split: train and calibrate on the 60 % of grids with the lowest mean reaction time, test on the top 20 %.
- SR-4 Two unit tests.
