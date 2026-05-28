import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

print("=" * 60)
print("METHOD 1: DIFFERENCE-IN-DIFFERENCES")
print("=" * 60)

# ── Part A: Classic Card & Krueger (NJ vs PA, 1992) ──────────────────────────
print("\nPart A: Card & Krueger Replication — NJ vs PA (1992)")
print("-" * 50)

nj_pa = panel[panel.state.isin(['NJ','PA'])].copy()
nj_pa['treated']  = (nj_pa['state'] == 'NJ').astype(int)
nj_pa['post']     = (nj_pa['year']  >= 1992).astype(int)
nj_pa['did']      = nj_pa['treated'] * nj_pa['post']

# Focus on 1990-1995 window
nj_pa_window = nj_pa[nj_pa['year'].between(1990, 1995)].copy()

# DiD regression
did_model = smf.ols('log_emp ~ treated + post + did', data=nj_pa_window).fit()

did_coef = did_model.params['did']
did_se   = did_model.bse['did']
did_t    = did_model.tvalues['did']
did_p    = did_model.pvalues['did']
did_ci   = did_model.conf_int().loc['did']

print(f"DiD Coefficient : {did_coef:.4f}")
print(f"Standard Error  : {did_se:.4f}")
print(f"t-statistic     : {did_t:.3f}")
print(f"p-value         : {did_p:.4f}")
print(f"95% CI          : [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
print(f"Interpretation  : NJ minimum wage increase associated with "
      f"{did_coef*100:.2f}% change in employment vs PA")

# Parallel trends test (pre-1992 only)
pre = nj_pa[nj_pa['year'].between(1990, 1991)].copy()
nj_pre = pre[pre.state=='NJ']['log_emp'].values
pa_pre = pre[pre.state=='PA']['log_emp'].values
print(f"\nParallel Trends Test (pre-1992):")
print(f"  NJ trend: {np.diff(nj_pre)[0]:+.4f}")
print(f"  PA trend: {np.diff(pa_pre)[0]:+.4f}")
print(f"  Difference: {(np.diff(nj_pre)-np.diff(pa_pre))[0]:+.4f} "
      f"({'Similar ✓' if abs((np.diff(nj_pre)-np.diff(pa_pre))[0]) < 0.05 else 'Different ✗'})")

# ── Part B: Two-Way Fixed Effects Panel DiD ───────────────────────────────────
print("\nPart B: Two-Way Fixed Effects Panel DiD (all 50 states)")
print("-" * 50)

# Create state and year dummies for TWFE
twfe_data = panel.copy()

# Treatment: log minimum wage (continuous)
twfe_data['log_mw'] = np.log(twfe_data['min_wage'])

# TWFE regression with state and year fixed effects
twfe_model = smf.ols(
    'log_emp ~ log_mw + C(state) + C(year)',
    data=twfe_data
).fit(cov_type='HC3')  # heteroscedasticity-robust SEs

mw_coef = twfe_model.params['log_mw']
mw_se   = twfe_model.bse['log_mw']
mw_p    = twfe_model.pvalues['log_mw']
mw_ci   = twfe_model.conf_int().loc['log_mw']

print(f"Log MW Coefficient (elasticity): {mw_coef:.4f}")
print(f"Standard Error (robust)         : {mw_se:.4f}")
print(f"p-value                         : {mw_p:.4f}")
print(f"95% CI                          : [{mw_ci[0]:.4f}, {mw_ci[1]:.4f}]")
print(f"Interpretation: 10% MW increase → {mw_coef*10:.2f}% employment change")
print(f"R-squared (within): {twfe_model.rsquared:.4f}")

# ── Part C: Event Study ───────────────────────────────────────────────────────
print("\nPart C: Event Study — NJ relative to PA around 1992")
print("-" * 50)

event_window = range(-3, 5)  # 3 years before to 4 years after
event_results = []

for k in event_window:
    year_k = 1992 + k
    if year_k in panel['year'].values:
        nj_emp = panel[(panel.state=='NJ') &
                       (panel.year==year_k)]['log_emp'].values[0]
        pa_emp = panel[(panel.state=='PA') &
                       (panel.year==year_k)]['log_emp'].values[0]
        # Normalize to pre-treatment level (1991)
        nj_base = panel[(panel.state=='NJ') &
                        (panel.year==1991)]['log_emp'].values[0]
        pa_base = panel[(panel.state=='PA') &
                        (panel.year==1991)]['log_emp'].values[0]
        relative_diff = (nj_emp - nj_base) - (pa_emp - pa_base)
        event_results.append({'k': k, 'diff': relative_diff})

event_df = pd.DataFrame(event_results)
print("Event study coefficients (relative to 1991 baseline):")
print(event_df.to_string(index=False))
pre_trend_coefs = event_df[event_df['k'] < 0]['diff'].abs().mean()
print(f"\nAverage pre-trend deviation: {pre_trend_coefs:.4f} "
      f"({'Clean ✓' if pre_trend_coefs < 0.02 else 'Noisy'})")

# ── Visualizations ────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "NJ vs PA Employment (log) — Card & Krueger Window",
        "Two-Way Fixed Effects: Log MW vs Log Employment",
        "Event Study — NJ relative to PA (1991=0)",
        "DiD Summary"
    ],
    specs=[[{"type":"scatter"},{"type":"scatter"}],
           [{"type":"scatter"},{"type":"bar"}]]
)

# Plot 1: NJ vs PA raw trends
colors = {'NJ':'#60a5fa','PA':'#f87171'}
for state in ['NJ','PA']:
    d = nj_pa[nj_pa.state==state].sort_values('year')
    fig.add_trace(go.Scatter(
        x=d['year'], y=d['log_emp'],
        mode='lines+markers', name=state,
        line=dict(color=colors[state], width=2.5),
        marker=dict(size=7)
    ), row=1, col=1)

fig.add_vline(x=1992, line_dash="dash", line_color="#f59e0b",
              line_width=2, row=1, col=1)
fig.add_annotation(x=1992, y=8.6, text="NJ raises MW<br>to $5.05",
                   font=dict(color="#f59e0b", size=10),
                   showarrow=False, row=1, col=1)

# Plot 2: TWFE scatter (log_mw vs log_emp residuals)
sample = twfe_data.sample(200, random_state=42)
fig.add_trace(go.Scatter(
    x=sample['log_mw'], y=sample['log_emp'],
    mode='markers', name='States',
    marker=dict(color='#60a5fa', size=5, opacity=0.5),
    showlegend=False
), row=1, col=2)
x_line = np.linspace(sample['log_mw'].min(), sample['log_mw'].max(), 100)
y_line = mw_coef * x_line + twfe_model.params['Intercept']
fig.add_trace(go.Scatter(
    x=x_line, y=y_line, mode='lines', name='TWFE fit',
    line=dict(color='#f59e0b', width=2.5), showlegend=False
), row=1, col=2)

# Plot 3: Event study
pre  = event_df[event_df['k'] < 0]
post = event_df[event_df['k'] >= 0]
fig.add_trace(go.Scatter(
    x=pre['k'], y=pre['diff'], mode='lines+markers',
    name='Pre-treatment', line=dict(color='#9ca3af', width=2),
    marker=dict(size=8, symbol='circle'),showlegend=False
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=post['k'], y=post['diff'], mode='lines+markers',
    name='Post-treatment', line=dict(color='#34d399', width=2),
    marker=dict(size=8, symbol='circle'), showlegend=False
), row=2, col=1)
fig.add_hline(y=0, line_dash="dot", line_color="#6b7280",
              line_width=1, row=2, col=1)
fig.add_vline(x=-0.5, line_dash="dash", line_color="#f59e0b",
              line_width=2, row=2, col=1)

# Plot 4: DiD summary bar
methods  = ['NJ vs PA DiD', 'TWFE Panel (×10%)']
coefs    = [did_coef, mw_coef * 0.10]
colors_b = ['#60a5fa' if c > 0 else '#f87171' for c in coefs]
fig.add_trace(go.Bar(
    x=methods, y=[c*100 for c in coefs],
    marker_color=colors_b,
    text=[f"{c*100:.2f}%" for c in coefs],
    textposition='outside', showlegend=False
), row=2, col=2)
fig.add_hline(y=0, line_color="#6b7280", line_width=1, row=2, col=2)

fig.update_layout(
    title="Method 1: Difference-in-Differences — Minimum Wage & Employment",
    template="plotly_dark", height=700,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color='white'), showlegend=True
)
fig.update_xaxes(gridcolor='#1f2937')
fig.update_yaxes(gridcolor='#1f2937')
fig.show()

print("\n" + "=" * 60)
print("DiD COMPLETE")
print("=" * 60)
