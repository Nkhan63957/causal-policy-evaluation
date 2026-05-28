from scipy.optimize import minimize
from plotly.subplots import make_subplots

print("=" * 60)
print("METHOD 3: SYNTHETIC CONTROL (REVISED)")
print("=" * 60)
print("Fix: Employment indexed to 1990=100 (removes scale bias)")
print("     Full pre-treatment path used as predictor matrix")
print("-" * 60)

TREATED_STATE  = 'CA'
TREATMENT_YEAR = 2014
PRE_YEARS      = list(range(1990, 2014))
POST_YEARS     = list(range(2014, 2024))
ALL_YEARS      = PRE_YEARS + POST_YEARS

# ── Donor pool ────────────────────────────────────────────────────────────────
federal_only = [
    'AL','GA','ID','IN','IA','KS','KY','LA','MS',
    'NC','ND','OK','SC','SD','TN','TX','VA','WY'
]
print(f"Donor pool: {len(federal_only)} federal-minimum-wage states")

# ── Index employment to 1990=100 per state ────────────────────────────────────
def get_indexed_series(state, years, base_year=1990):
    """Employment indexed to base_year=100."""
    base_val = panel[(panel.state==state) &
                     (panel.year==base_year)]['employment'].values
    if len(base_val) == 0 or base_val[0] == 0:
        return np.full(len(years), np.nan)
    base_val = base_val[0]
    vals = []
    for y in years:
        row = panel[(panel.state==state) & (panel.year==y)]
        if len(row) > 0:
            vals.append(row['employment'].values[0] / base_val * 100)
        else:
            vals.append(np.nan)
    return np.array(vals)

# Full pre-treatment path as predictors
Y_1_pre = get_indexed_series(TREATED_STATE, PRE_YEARS)

Y_0_pre = np.column_stack([
    get_indexed_series(s, PRE_YEARS)
    for s in federal_only
])

print(f"CA indexed pre-period: {Y_1_pre[[0,5,10,15,20,23]].round(1)}")
print(f"  (1990=100, 2013={Y_1_pre[-1]:.1f})")

# ── Optimize weights ──────────────────────────────────────────────────────────
n_donors = len(federal_only)

def objective(W):
    synth = Y_0_pre @ W
    return np.sum((Y_1_pre - synth)**2)

W0 = np.ones(n_donors) / n_donors
result = minimize(
    objective, W0,
    method='SLSQP',
    bounds=[(0,1)]*n_donors,
    constraints=[{'type':'eq','fun': lambda W: np.sum(W)-1}],
    options={'ftol':1e-12,'maxiter':2000}
)

W_opt = np.maximum(result.x, 0)
W_opt /= W_opt.sum()

print(f"\nOptimization: {'Converged ✓' if result.success else 'Did not converge'}")

weight_df = pd.DataFrame({
    'state': federal_only, 'weight': W_opt
}).sort_values('weight', ascending=False)
print("\nTop donor weights:")
print(weight_df[weight_df['weight'] > 0.005].to_string(index=False))

# ── Build synthetic CA for full period ───────────────────────────────────────
real_ca_idx   = get_indexed_series(TREATED_STATE, ALL_YEARS)
synth_ca_idx  = sum(W_opt[j] * get_indexed_series(s, ALL_YEARS)
                    for j, s in enumerate(federal_only))
gap_idx       = real_ca_idx - synth_ca_idx

pre_idx  = list(range(len(PRE_YEARS)))
post_idx = list(range(len(PRE_YEARS), len(ALL_YEARS)))

pre_mspe  = np.mean(gap_idx[pre_idx]**2)
post_mspe = np.mean(gap_idx[post_idx]**2)
mspe_ratio = post_mspe / pre_mspe if pre_mspe > 0 else 0
att        = np.mean(gap_idx[post_idx])

print(f"\nFit quality:")
print(f"  Pre-treatment MSPE  : {pre_mspe:.4f}")
print(f"  Post-treatment MSPE : {post_mspe:.4f}")
print(f"  MSPE ratio          : {mspe_ratio:.2f}x")
print(f"  ATT (avg gap)       : {att:+.2f} index points")
print(f"  Interpretation: CA employment {att:+.1f} points vs synthetic "
      f"({'above' if att>0 else 'below'} counterfactual)")

# ── Placebo tests ─────────────────────────────────────────────────────────────
print(f"\nRunning {len(federal_only)} placebo tests...")

placebo_gaps   = []
placebo_ratios = []

for placebo_state in federal_only:
    donors_p = [s for s in federal_only if s != placebo_state]
    Y_p_pre  = get_indexed_series(placebo_state, PRE_YEARS)
    Y_d_pre  = np.column_stack([get_indexed_series(s, PRE_YEARS)
                                 for s in donors_p])
    n_d = len(donors_p)
    res_p = minimize(
        lambda W: np.sum((Y_p_pre - Y_d_pre@W)**2),
        np.ones(n_d)/n_d, method='SLSQP',
        bounds=[(0,1)]*n_d,
        constraints=[{'type':'eq','fun':lambda W:np.sum(W)-1}],
        options={'ftol':1e-10,'maxiter':1000}
    )
    W_p = np.maximum(res_p.x,0); W_p /= W_p.sum()

    real_p  = get_indexed_series(placebo_state, ALL_YEARS)
    synth_p = sum(W_p[j]*get_indexed_series(s,ALL_YEARS)
                  for j,s in enumerate(donors_p))
    gap_p   = real_p - synth_p

    pre_m   = np.mean(gap_p[pre_idx]**2)
    post_m  = np.mean(gap_p[post_idx]**2)

    # Keep only placebos with good pre-treatment fit
    if pre_m < 5 * pre_mspe and pre_m > 0:
        placebo_gaps.append(gap_p)
        placebo_ratios.append(post_m/pre_m)

pseudo_p = np.mean([r >= mspe_ratio for r in placebo_ratios])
print(f"Valid placebos: {len(placebo_gaps)}")
print(f"CA MSPE ratio : {mspe_ratio:.2f}x")
print(f"Pseudo p-value: {pseudo_p:.3f} "
      f"({'Significant ✓' if pseudo_p<0.10 else 'Not significant at 10%'})")

# ── Visualization ─────────────────────────────────────────────────────────────
fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Real vs Synthetic California (Index 1990=100)",
        "Treatment Gap (Real − Synthetic)",
        "Placebo Tests — Donor States (grey) vs CA (blue)",
        "MSPE Ratio Distribution — Placebo Inference"
    ]
)

# Plot 1: Real vs synthetic
fig3.add_trace(go.Scatter(
    x=ALL_YEARS, y=real_ca_idx, mode='lines',
    name='Real California',
    line=dict(color='#60a5fa',width=3)
), row=1,col=1)
fig3.add_trace(go.Scatter(
    x=ALL_YEARS, y=synth_ca_idx, mode='lines',
    name='Synthetic California',
    line=dict(color='#f59e0b',width=2.5,dash='dash')
), row=1,col=1)
fig3.add_vrect(x0=TREATMENT_YEAR,x1=2023,
               fillcolor='rgba(248,113,113,0.08)',
               line_width=0,row=1,col=1)
fig3.add_vline(x=TREATMENT_YEAR,line_dash="dash",
               line_color="#f87171",line_width=2,row=1,col=1)
fig3.add_annotation(x=2018,y=real_ca_idx.max()*0.88,
                    text="MW increases<br>begin 2014",
                    font=dict(color="#f87171",size=10),
                    showarrow=False,row=1,col=1)

# Plot 2: Gap
fig3.add_trace(go.Scatter(
    x=ALL_YEARS, y=gap_idx, mode='lines+markers',
    name='Gap', showlegend=False,
    line=dict(color='#34d399',width=2.5),
    marker=dict(size=5)
), row=1,col=2)
fig3.add_hline(y=0,line_dash="dot",
               line_color="#6b7280",line_width=1.5,row=1,col=2)
fig3.add_vline(x=TREATMENT_YEAR,line_dash="dash",
               line_color="#f87171",line_width=2,row=1,col=2)
fig3.add_annotation(
    x=2019,
    y=att+5 if att>0 else att-5,
    text=f"ATT = {att:+.1f} pts",
    font=dict(color="#34d399",size=11),
    showarrow=False,row=1,col=2
)

# Shade pre/post
pre_y  = gap_idx[pre_idx]
post_y = gap_idx[post_idx]
fig3.add_trace(go.Scatter(
    x=PRE_YEARS+PRE_YEARS[::-1],
    y=list(pre_y)+[0]*len(PRE_YEARS),
    fill='toself',fillcolor='rgba(148,163,184,0.08)',
    line_width=0,showlegend=False
), row=1,col=2)

# Plot 3: Placebo
for pg in placebo_gaps:
    fig3.add_trace(go.Scatter(
        x=ALL_YEARS, y=pg, mode='lines',
        line=dict(color='rgba(148,163,184,0.2)',width=1),
        showlegend=False
    ), row=2,col=1)
fig3.add_trace(go.Scatter(
    x=ALL_YEARS, y=gap_idx, mode='lines',
    name='California', showlegend=True,
    line=dict(color='#60a5fa',width=3)
), row=2,col=1)
fig3.add_hline(y=0,line_dash="dot",
               line_color="#6b7280",line_width=1,row=2,col=1)
fig3.add_vline(x=TREATMENT_YEAR,line_dash="dash",
               line_color="#f87171",line_width=2,row=2,col=1)

# Plot 4: MSPE histogram
fig3.add_trace(go.Histogram(
    x=placebo_ratios, nbinsx=12,
    marker_color='#374151',
    marker_line=dict(color='#60a5fa',width=1),
    showlegend=False
), row=2,col=2)
fig3.add_vline(x=mspe_ratio,line_dash="dash",
               line_color="#60a5fa",line_width=2.5,row=2,col=2)
fig3.add_annotation(
    x=mspe_ratio, y=2,
    text=f"CA: {mspe_ratio:.1f}x",
    font=dict(color="#60a5fa",size=11),
    showarrow=True,arrowcolor="#60a5fa",
    ax=40,row=2,col=2
)

fig3.update_layout(
    title="Method 3: Synthetic Control — California MW Increases (2014+)",
    template="plotly_dark",height=700,
    paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
    font=dict(color='white')
)
fig3.update_xaxes(gridcolor='#1f2937')
fig3.update_yaxes(gridcolor='#1f2937')
fig3.update_xaxes(title_text="Year",row=1,col=1)
fig3.update_xaxes(title_text="Year",row=1,col=2)
fig3.update_xaxes(title_text="Year",row=2,col=1)
fig3.update_xaxes(title_text="MSPE Ratio",row=2,col=2)
fig3.update_yaxes(title_text="Employment Index (1990=100)",row=1,col=1)
fig3.update_yaxes(title_text="Gap (index points)",row=1,col=2)
fig3.update_yaxes(title_text="Gap (index points)",row=2,col=1)
fig3.update_yaxes(title_text="Count",row=2,col=2)
fig3.show()

print("\n" + "="*60)
print("SYNTHETIC CONTROL COMPLETE")
print("="*60)
