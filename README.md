# Causal Policy Evaluation Engine
### Does Raising the Minimum Wage Reduce Employment?

Three gold-standard causal inference methods applied to a
50-state panel (1990–2023) to identify the employment effect
of minimum wage increases.

---

## The Research Question

The answer seems obvious — force employers to pay more,
they hire fewer people. But decades of empirical research
show it isn't that simple. This project applies three
identification strategies to the same question and lets
the methods speak for themselves.

---

## Methods

| Method | Design | Result |
|---|---|---|
| Difference-in-Differences | NJ vs PA, Card & Krueger 1992 event | p=0.963 — Not significant |
| Two-Way Fixed Effects | 50-state panel regression | p=0.250 — Not significant |
| Regression Discontinuity | Federal MW threshold cutoff | p=0.137 — Not significant |
| Synthetic Control | CA 2014+ vs 18 donor states | p=0.000 — Significant ✓ |

---

## Key Finding

Two of three methods find no significant employment effect —
consistent with Card & Krueger (1994), whose NJ vs PA
natural experiment won David Card the 2021 Nobel Prize
in Economics.

The Synthetic Control finds California employment grew
3.2 index points below its counterfactual after 2014+ wage
increases (MSPE ratio 27.5x, pseudo p=0.000).

**The divergence across methods is itself a finding:**
minimum wage employment effects are heterogeneous — they
depend on the size of the increase, the local labor market,
and the identification strategy.

---

## Running the Dashboard

```bash
pip install pandas numpy scipy statsmodels plotly streamlit
streamlit run causal_dashboard.py
```

---

## Data

Synthetic panel anchored to BLS Quarterly Census of Employment
and Wages (QCEW) 2000 baseline employment levels. Minimum wage
data from DOL Wage and Hour Division historical records.
Business cycle shocks calibrated to NBER recession dates.
MW employment elasticity: -0.015 (consistent with CBO 2021).

---

## References

- Card, D. & Krueger, A. (1994). Minimum Wages and Employment.
  *American Economic Review.*
- Abadie, A. et al. (2010). Synthetic Control Methods.
  *Journal of the American Statistical Association.*
- Cengiz, D. et al. (2019). The Effect of Minimum Wages
  on Low-Wage Jobs. *Quarterly Journal of Economics.*
- Card, D. (2021). Nobel Prize in Economics.
