# Assignment 1 Report

- **Name**: Jesús López Cubas
- **Student ID**:
- **Email**: jlopez.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

---

**TO:** Portfolio Manager, Index Volatility Book (hypothetical multi-manager pod)
**FROM:** Jesús López Cubas, Analyst
**RE:** A one-month "vol will rise" signal for sizing the book
**DATE:** 11 October 2026

**Bottom line.** The model ranks the months in which our sizing will underestimate risk far better than the rule we use today: out of sample (2020 to 2026) its AUC is 0.79 against 0.65 for the mean reversion rule, and at the cost structure we agreed it cuts the cost of sizing errors from 17.8 to 13.2 bps per decision. Three independent implementations (scikit-learn, a hand-written PyTorch loop and the standard PyTorch workflow) land on the same solution, with predicted probabilities that differ by at most 0.00005 and identical decisions on all 1,681 test days. **I do not recommend running it as an on/off overlay yet.** At a 3x cost on misses it tells us to cut on 83% of days, and over 81 test months that behaved like simply running 25% less risk, with a slightly lower Sharpe (0.67 against 0.71). The value is in the ranking, and the next version should size continuously on it rather than switch on a threshold.

---

## Dataset

**What it is.** Daily S&P 500 index prices (open, high, low, close, volume) and the daily VIX index, downloaded from Yahoo Finance with `download_data.py` and committed to `data/raw/` so that training never touches the network and anyone can retrain on exactly the same data. One row is one US trading day. The raw files have 9,261 S&P rows and 9,263 VIX rows from 2 January 1990 to 9 October 2026, under 1MB in total. After joining and building features, the modelling table has 9 features, a binary target and a few helper columns (trailing and future realized vol, leverage, next-month fund return) used for the baselines and the business cost.

**Why this dataset.** Volatility is the one quantity in equity markets that is genuinely forecastable, because it clusters and mean-reverts, and sizing on it is the core risk decision of a pod. It also forces the three things this assignment is about. The target does not exist in the data and has to be built, the series is a time series where a careless split leaks the future, and the decision threshold has an obvious business cost behind it.

## Business / real-life framing

**The book and the problem.** The book runs a volatility target: it holds the S&P 500 at a leverage of 10% divided by trailing 21-day realized vol, capped at 2x, and resets that sizing monthly. The sizing silently assumes next month will look like last month. When next month turns out more volatile, the book runs above its risk target, and in a pod that is how drawdown limits get hit. The decision this model supports is therefore simple and taken every day: **should we cut gross exposure by 25% for the next month?**

**Target definition.** The label is 1 when realized vol over the next 21 trading days ends up above the trailing 21-day vol the book is sized on, and 0 otherwise. I chose this relative definition over an absolute one ("next month's vol above 20%") for two reasons found on the training years. First, the absolute question is mostly answered already by the sizing itself: the rule "high vol stays high" gets 0.86 accuracy on its own, against 0.80 for "always calm", so a model would add little. Second, the absolute base rate moves with the era (9% of days in the 1990s, 34% in the 2000s), while the relative one sits between 46% and 49% in every block. The horizon is 21 days because that is the window the book sizes on and holds. A 5-day version would give more independent observations, but each side of the comparison would rest on five returns, and the strongest signal (the VIX gap) drops from an AUC of 0.75 to 0.60.

**Split.** A random split is not an option. Day t and day t+1 share 20 of their 21 future returns, so their labels are nearly identical, and a random split would put near-duplicates on both sides and flatter every metric. The data is split into three fixed chronological blocks: **train 1990 to 2014** (6,028 days), **validation 2015 to 2019** (1,237 days) and **test 2020 to 2026** (1,681 days). A day stays in a block only if its whole 21-day label window ends inside that block, which purges the last 21 days before each boundary (42 days in total). Validation is used for every choice (regularization, learning rate, threshold, feature selection) and the test block is scored once. The test years are a genuine stress test: the COVID crash, the 2022 bear market and the April 2025 tariff shock. I used one fixed split rather than walk-forward retraining because the point is to compare three implementations on identical data, and I did not refit on train plus validation after tuning, because the threshold was chosen for the model trained on 1990 to 2014 and should be applied to that same model.

**Which metric sets the threshold.** Not accuracy and not F1, but business cost. A **false alarm** (cutting in a month where vol did not rise) gives up 25% of the book's next-month return in a calm month: **29.4 bps**, measured on the training years. A **miss** is harder to price, and the data shows why. On average P&L a miss costs nothing: the book's average return in vol-rising months is 0.006%, so a haircut there saves 0.15 bps on average. But 97% of the book's worst 5% of months are vol-rising months, the probability of losing more than 5% in a month is 7.2% when vol rises against 0.1% when it does not, and the haircut cuts that probability to 3.1%. The haircut is insurance: it never pays on average and it pays exactly when a drawdown limit is at stake. I therefore price a miss at **3x a false alarm (88.3 bps)** as an explicit risk-appetite assumption, show the sensitivity to that multiple below, and expose it as a slider on the dashboard. The threshold is the probability that minimizes total cost on validation. Regularization, by contrast, is tuned on validation log loss, because it should make the probabilities good, while the threshold turns probabilities into decisions.

**Baselines.** Two, for different purposes. **Never cut** (the majority class) is the floor and only shows the model learned something. The **PM's mean reversion rule** ("vol will rise if this month's vol is below its 1-year average") is the bar: it is what the desk would do without a model, so the model only earns its place if it beats it. On the training years this rule scores 0.64 accuracy, while momentum ("vol rose last month, so it keeps rising") scores 0.39, worse than a coin flip, which is the first sign that this target is about mean reversion.

## Data preparation & feature engineering

**Data-quality decisions.** Four problems turned up when checking the raw files, each handled explicitly in `volmodel/data.py`. The S&P **open is not real** for much of the sample: it equals the previous close on 84% of days from 1990 to 2005 and 15% of days from 2006 to 2013, so the open is dropped and no feature uses it (no overnight gap, no Garman Klass estimator). High and low are trustworthy, which I checked by comparing the Parkinson high-low variance to close-to-close variance year by year: the ratio sits between 0.54 and 0.90 in every year except 2020 (0.37, when overnight gaps dominated), with no break in the early years. The **VIX open, high and low are flat** on all 506 trading days of 1990 and 1991, so only the VIX close is used. Two **VIX rows fall on US holidays** with no S&P trading (25 May and 7 September 2026) and are dropped by the join. Finally, the last 21 days have no label yet: they are kept as "live" rows, predicted on the dashboard and never scored, and the first 252 days are lost to the 1-year features.

**Features.** Every feature uses only information available at the close of day t. A test in `tests/test_pipeline.py` enforces this: features computed on data cut at a date are identical to features computed on the full history.

| Feature | Definition | Why it should help |
|---|---|---|
| `log_rv_1d`, `log_rv_5d`, `log_rv_21d` | Log annualized realized vol over the last 1, 5 and 21 days | Vol has short and long memory at once (the HAR structure of Corsi, 2009). Logs because vol is right-skewed |
| `log_rv_21d_vs_1y` | Log of 21-day vol over 1-year vol | Distance from the long-run level: the continuous version of the mean reversion rule |
| `vix_gap` | Log VIX minus log 21-day realized vol | Risk the options market is pricing that has not shown up in returns yet |
| `vix_change_5d` | 5-day change in log VIX | Implied vol moving ahead of realized vol |
| `range_vs_close_5d` | Log of Parkinson high-low vol over close-to-close vol, last 5 days | Intraday turbulence that closing prices miss |
| `return_21d` | S&P log return over the last 21 days | Leverage effect: selloffs feed volatility more than rallies |
| `drawdown_1y` | Log distance from the 1-year closing high | Volatility rises in drawdowns |

**Preprocessing.** Each feature is clipped to its 1st and 99th percentiles on the training years, then standardized with the training mean and standard deviation. No statistic is ever computed on validation or test. Clipping matters here because the test years contain COVID, where several features sit far outside anything seen in training, and a linear model would turn those values straight into extreme probabilities. Standardizing matters for two reasons: gradient descent in PyTorch converges badly when features live on very different scales, and the L2 penalty treats all coefficients alike, which is only fair if the features share a scale. All three implementations receive exactly the same float32 matrix.

**What worked and what did not.** I built 11 candidate features and kept 9. The table adds feature groups one at a time and reports validation scores with the exact solver.

| Feature group added | Validation AUC | Validation log loss |
|---|---|---|
| HAR realized vol (1d, 5d, 21d) | 0.728 | 0.630 |
| + Mean reversion (21d vs 1y vol) | 0.731 | 0.623 |
| + VIX (gap to realized, 5d change) | 0.787 | 0.558 |
| + Intraday range (Parkinson vs close) | 0.790 | 0.549 |
| + Window shape (spike share, down share) | 0.789 | 0.551 |
| + Price path (21d return, 1y drawdown) | 0.791 | 0.551 |

The **VIX group does the heavy lifting**: it adds 0.055 of AUC and cuts log loss from 0.623 to 0.558, and removing it from the full model costs 0.027 of AUC, more than any other group. The **intraday range** adds a little on top. The **window shape** features did not work. The idea was that when one large day inflated trailing vol, it will drop out of the window and next month will look calmer (the spike share), and that down-day variance matters more (the down share). Adding them lowered validation AUC and raised log loss, and their coefficients were close to zero, because the 1-day and 5-day vol terms and the 21-day return already carry that information. I dropped them. One honest caveat on this table: removing the HAR group or the mean reversion term from the full candidate model slightly *raises* validation AUC (to 0.793 and 0.795). Those terms overlap by construction (`log_rv_21d_vs_1y` is `log_rv_21d` minus 1-year vol), and with roughly 59 independent months in validation, differences of 0.005 in AUC are within noise. I kept them because their coefficients are large and economically meaningful, and dropped only the features that showed no signal on either measure.

**What the model learned.** On standardized features, the largest coefficient is the **VIX gap (+0.91)**: when options price more risk than has been realized, vol tends to rise. Trailing 21-day vol (**−0.42**) and its distance from the 1-year level (**−0.24**) carry mean reversion. Five-day vol is positive (**+0.27**), so vol accelerating within the month pushes the probability up. The 21-day return is negative (**−0.26**), the leverage effect. The intraday range (+0.17), the drawdown (−0.10), the VIX change (+0.08) and 1-day vol (+0.04) play smaller roles. Every sign matches the economic story, and the three implementations agree on all of them.

## Modeling: three implementations, one model

**Model.** L2-regularized logistic regression, because the target is binary and the decision is a probability compared against a cost-based threshold. All three implementations minimize the same objective: mean binary cross-entropy plus (λ/2)‖w‖², with the intercept unpenalized. scikit-learn writes this as C × Σ cross-entropy + ½‖w‖², so dividing by C × n gives **λ = 1 / (C × n)**. That conversion is what lets the three agree.

**The three implementations** (`volmodel/models.py`). **scikit-learn** solves the problem with L-BFGS to a tolerance of 1e-10 and serves as the reference solution. The **manual PyTorch loop** follows Session 5: raw tensors for the weights and the intercept (starting at zero), the cross-entropy written from its formula in its numerically stable form, `loss.backward()` to get gradients, a hand-written update `w -= lr * w.grad` inside `torch.no_grad()`, and gradients reset after every step. The **standard workflow** uses an `nn.Module` with one `nn.Linear` layer (random initialization with a fixed seed), `BCEWithLogitsLoss`, and `torch.optim.SGD` with `weight_decay = λ` applied to the weights only, which adds exactly λw to the gradient. Both PyTorch versions use full-batch gradient descent for 3,000 epochs.

**Tuning, sparingly.** C was chosen from {0.001, 0.01, 0.1, 1, 10, 100} on validation log loss: **C = 0.01** (log loss 0.553 against 0.561 for C = 1), so λ = 0.0166. Moderate regularization helps because the HAR terms are strongly correlated. C was tuned once with the exact solver and passed to PyTorch through λ, because it is a property of the model, not of the solver. The learning rate only changes how fast gradient descent reaches the optimum, not the optimum itself, so I judged it by the distance to the scikit-learn objective after 3,000 epochs. At 0.01 both PyTorch versions were still 5e-4 away from the optimum. At 0.1, 0.5 and 1.0 they reached it within float32 rounding, and I took **lr = 0.1**, the smallest rate that converged.

**Results on the test block (2020 to 2026), never seen during any choice.**

| Model | AUC | Log loss | Brier | Accuracy | Precision | Recall | Days flagged | Cost per decision (bps) |
|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.7946 | 0.5477 | 0.1852 | 0.603 | 0.553 | 0.947 | 83% | 13.2 |
| PyTorch manual loop | 0.7946 | 0.5477 | 0.1852 | 0.603 | 0.553 | 0.947 | 83% | 13.2 |
| PyTorch nn.Module + optim | 0.7946 | 0.5477 | 0.1852 | 0.603 | 0.553 | 0.947 | 83% | 13.2 |
| Baseline: never cut (majority class) | 0.5000 | 0.6927 | 0.2498 | 0.515 | n/a | 0.000 | 0% | 42.8 |
| Baseline: PM mean reversion rule | 0.6545 | n/a | n/a | 0.608 | 0.570 | 0.779 | 66% | 17.8 |
| Policy: always cut | n/a | n/a | n/a | n/a | n/a | 1.000 | 100% | 15.2 |

On validation the picture is the same: AUC 0.790 for all three against 0.708 for the rule, and 12.6 bps per decision against 17.6 for the rule. Accuracy is not the right lens here. The rule's accuracy is about the same as the model's, but the model's errors are the cheap kind (false alarms) and the rule's are the expensive kind (misses: 180 against 43).

**Do the three agree? Yes, to float32 precision.** All three reach the same training objective, 0.543864464 to the ninth decimal. On the 1,681 test days their probabilities differ by at most **3.1e-5** (scikit-learn against either PyTorch version) and 4.4e-5 between the two PyTorch versions, and they make **identical decisions on every day**. The remaining differences have three sources: PyTorch works in float32 and scikit-learn in float64, gradient descent stops after a finite number of steps, and the two PyTorch versions start from different points (zeros and random weights). That last point is itself a check: logistic regression is convex, so both starting points end at the same minimum. Two observations show that this agreement is earned, not automatic. With the 0.01 learning rate the PyTorch versions had not converged and their coefficients were visibly off. And with the original 11 features the largest probability gap was ten times bigger (3.7e-4, with one day decided differently), because the redundant features flattened the objective along some directions, so the solvers stopped at slightly different coefficients with the same loss. Dropping them improved the conditioning of the problem.

**The threshold.** The cost-minimizing threshold on validation is **0.16**. With perfectly calibrated probabilities, theory says the optimal threshold is 1 / (1 + 3) = 0.25. The model is reasonably calibrated on test (days scored below 0.2 saw vol rise 15% of the time, days above 0.8 saw it 89% of the time), and the validation cost curve is flat between the two (12.6 bps at 0.16, 13.1 at 0.25). On test the theoretical threshold would have done slightly better (12.8 bps). I kept the validation choice, since switching after looking at test results would be using the test set to make a decision.

**How much the conclusion depends on the miss cost.** The threshold is always re-chosen on validation, and the cost is measured on test, in bps per decision.

| Miss cost (multiple of a false alarm) | Threshold | Days flagged | Model | Always cut | Never cut | Mean reversion rule |
|---|---|---|---|---|---|---|
| 0.5x | 0.82 | 13% | 5.8 | 15.2 | 7.1 | 10.0 |
| 1x | 0.47 | 47% | 8.1 | 15.2 | 14.3 | 11.5 |
| 1.5x | 0.42 | 51% | 10.4 | 15.2 | 21.4 | 13.1 |
| 2x | 0.42 | 51% | 12.3 | 15.2 | 28.5 | 14.7 |
| 3x | 0.16 | 83% | 13.2 | 15.2 | 42.8 | 17.8 |
| 5x | 0.16 | 83% | 14.7 | 15.2 | 71.4 | 24.2 |

The model is the cheapest policy at every multiple, but its edge over "always cut" shrinks as misses get more expensive: at 3x it flags most days and saves only 2 bps per decision against cutting permanently. The model is most useful where the decision is genuinely two-sided, between 1x and 2x.

**What it would have done to the book.** A monthly-rebalanced backtest on the test block, with one sizing decision every 21 days (81 months), S&P price return only and no costs:

| Strategy | Annual return | Annual vol | Sharpe | Max drawdown | Worst month | Months below −5% |
|---|---|---|---|---|---|---|
| Vol target only | 8.2% | 12.2% | 0.71 | −14.9% | −9.7% | 4 |
| Always 25% less risk | 6.2% | 9.1% | 0.71 | −11.3% | −7.3% | 1 |
| With model overlay (cut 68 of 81 months) | 6.1% | 9.6% | 0.67 | −13.4% | −7.3% | 2 |
| With mean reversion rule overlay (cut 55 of 81) | 6.5% | 10.0% | 0.68 | −13.4% | −7.3% | 4 |

This is the uncomfortable result. The overlay cut the tail, with half as many months below −5% as the unhedged book, but a permanent 25% cut did as well or better on every risk measure with the same Sharpe. Two episodes explain it. In **July 2024** trailing vol was 6%, so the book was running at 1.6x leverage, and the model put the probability of a vol rise at 0.88. It was right: the August yen carry unwind took the book down 9.7% at full size, 7.3% after the haircut. In **March 2020** trailing vol was already 34% and VIX 32, and the model put the probability at 0.13, betting on mean reversion. Vol went to 93%. The low sizing limited the loss to 5.7%, but this is the failure mode of a model built on mean reversion: it misses escalation when a shock is already under way. With 81 months and a handful of episodes deciding the result, none of these backtest differences is statistically meaningful, and I would not draw a P&L conclusion from them in either direction.

## Limitations & next steps

**The label ignores magnitude.** A month where vol ends 1% above trailing counts the same as March 2020, where it nearly tripled, and the cost function treats both misses alike. This is the main reason the classification results look better than the backtest. Next step: model the log ratio of next-month to trailing vol with a linear regression (the same pipeline with a continuous target), or weight training days by the size of the overshoot, and size the book continuously on the forecast instead of switching at a threshold.

**Escalating crises are the blind spot.** The coefficients are dominated by mean reversion, so the model expects high vol to fall, which is usually right and catastrophically wrong in March 2020. Next step: add features that separate a shock that is fading from one that is spreading, such as the VIX term structure (VIX against 3-month VIX, available from 2007) and credit spreads from FRED, and test them on the same validation block.

**The 3x miss cost is a judgment, not an estimate.** The sensitivity table shows the conclusion depends on it. Next step: derive it from the pod's actual drawdown rules with the risk team, as the probability of hitting a stop multiplied by the capital at stake, instead of choosing a round number.

**The effective sample is small.** Overlapping labels mean roughly 59 independent months in validation and 81 in test. Differences of a few thousandths of AUC, and all the backtest differences, are within noise. Next step: block-bootstrap confidence intervals on the metrics and a walk-forward evaluation across 2000 to 2026 to put a range around every number in this memo.

**One split, no retraining.** In production the model would be retrained as data arrives. Next step: annual walk-forward retraining with monitoring of calibration drift, since the threshold assumes the probabilities keep their meaning.

**A simplified book and imperfect data.** The backtest uses price returns without dividends, financing or trading costs, rebalances only monthly, and relies on Yahoo data whose open prices proved unreliable. Next step: a total-return index from a professional vendor and a cost charged on every haircut.

I am happy to walk through any of this, and the dashboard (`uv run python main.py app`) lets you move the threshold and the miss cost and watch the confusion matrix, the cost and the backtest respond.

J. López Cubas, Madrid

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment, with my professor's explicit approval by email on 11 October 2026, on the condition that the decisions are mine and the AI may write. Claude proposed options for each decision with supporting evidence from the data (dataset and topic, target definition, horizon, split design, baselines, cost structure and feature list), wrote the code (data pipeline, the three implementations, tests and the Gradio dashboard) and drafted this report from the results. I made each decision: the stock market volatility topic, the relative target, the 21-day horizon, the three fixed blocks without refitting, the two baselines, the 25% haircut with a 3x miss cost, the feature list, and dropping the two window shape features after the ablation. For several of these (the refit question, the baselines, the cost option, scripts over notebooks and the dropped features) I asked for Claude's recommendation and followed it after reviewing the reasoning. I have reviewed every section of this report and the code, and I can explain each decision without them.
