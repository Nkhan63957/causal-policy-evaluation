print("=" * 60)
print("METHOD 2: REGRESSION DISCONTINUITY")
print("=" * 60)

# ── Setup ─────────────────────────────────────────────────────────────────────
# Running variable: percentage by which state MW exceeds federal MW
# Cutoff: 0 (exactly at federal minimum)
# Outcome: log employment growth (year-over-year)
# Design: sharp RD — states either at federal or above

rd_data = panel.copy()

# Running variable: % above federal MW
rd_data['fed_mw'] = rd_data['year'].map(FED_MW)
rd_data['mw_gap'] = (rd_data['min_wage'] - rd_data['fed_mw']) / rd_data['fed_mw']

# Employment growth (year over year, within state)
rd_data = rd_data.sort_values(['state','year'])
rd_data['emp_growth'] = rd_data.groupby('state')['log_emp'].diff()

# Treatment indicator: strictly above federal
rd_data['above_cutoff'] = (rd_data['mw_gap'] > 0).astype(int)

# Remove the year when federal MW itself jumps (confounds the design)
fed_jump_years = [1996, 1997, 2007, 2008, 2009]
rd_clean = rd_data[
    (~rd_data['year'].isin(fed_jump_years)) &
    (rd_data['emp_growth'].notna()) &
    (rd_data['mw_gap'].between(-0.30, 0.30))  # bandwidth: ±30% of federal
].copy()

print(f"RD sample: {len(rd_clean)} observations")
print(f"Bandwidth: ±30% of federal minimum wage")
print(f"Above cutoff: {rd_clean['above_cutoff'].sum()} obs")
print(f"Below/at cutoff: {(rd_clean['above_cutoff']==0).sum()} obs")

# ── Local Linear Regression on each side of cutoff ───────────────────────────
# Fit separate linear regressions on each side
# The discontinuity at cutoff = 0 is the RD estimate

left  = rd_clean[rd_clean['above_cutoff'] == 0].copy()
right = rd_clean[rd_clean['above_cutoff'] == 1].copy()

# Left side: states at federal minimum (mw_gap <= 0)
left_model  = smf.ols('emp_growth ~ mw_gap', data=left).fit()
# Right side: states above federal minimum (mw_gap > 0)
right_model = smf.ols('emp_growth ~ mw_gap', data=right).fit()

# RD estimate: difference in intercepts at cutoff (mw_gap = 0)
left_intercept  = left_model.params['Intercept']
right_intercept = right_model.params['Intercept']
rd_estimate     = right_intercept - left_intercept

# Standard error via pooled regression with interaction
rd_clean['above_x_gap'] = rd_clean['above_cutoff'] * rd_clean['mw_gap']
rd_pooled = smf.ols(
    'emp_growth ~ above_cutoff + mw_gap + above_x_gap',
    data=rd_clean
).fit(cov_type='HC3')

rd_coef = rd_pooled.params['above_cutoff']
rd_se   = rd_pooled.bse['above_cutoff']
rd_t    = rd_pooled.tvalues['above_cutoff']
rd_p    = rd_pooled.pvalues['above_cutoff']
rd_ci   = rd_pooled.conf_int().loc['above_cutoff']

print(f"\nRD Estimate (discontinuity at cutoff):")
print(f"  Left intercept  : {left_intercept:.6f}")
print(f"  Right intercept : {right_intercept:.6f}")
print(f"  Discontinuity   : {rd_estimate:.6f}")
print(f"\nPooled RD regression:")
print(f"  Coefficient : {rd_coef:.6f}")
print(f"  Std Error   : {rd_se:.6f}")
print(f"  t-statistic : {rd_t:.3f}")
print(f"  p-value     : {rd_p:.4f}")
print(f"  95% CI      : [{rd_ci[0]:.6f}, {rd_ci[1]:.6f}]")
print(f"  Interpretation: Crossing the federal MW threshold associated with "
      f"{rd_coef*100:.3f}pp change in employment growth")

# ── Bandwidth Sensitivity ─────────────────────────────────────────────────────
print(f"\nBandwidth Sensitivity:")
print(f"{'Bandwidth':>12} {'Coef':>10} {'SE':>10} {'p-value':>10} {'N':>8}")
print("-" * 55)

for bw in [0.10, 0.15, 0.20, 0.25, 0.30]:
    bw_data = rd_data[
        (~rd_data['year'].isin(fed_jump_years)) &
        (rd_data['emp_growth'].notna()) &
        (rd_data['mw_gap'].between(-bw, bw))
    ].copy()
    bw_data['above_x_gap'] = bw_data['above_cutoff'] * bw_data['mw_gap']
    try:
        m = smf.ols('emp_growth ~ above_cutoff + mw_gap + above_x_gap',
                    data=bw_data).fit(cov_type='HC3')
        print(f"  ±{bw*100:.0f}%        {m.params['above_cutoff']:>10.6f} "
              f"{m.bse['above_cutoff']:>10.6f} "
              f"{m.pvalues['above_cutoff']:>10.4f} "
              f"{len(bw_data):>8}")
    except:
        print(f"  ±{bw*100:.0f}%        insufficient data")

# ── Visualization ─────────────────────────────────────────────────────────────
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "RD Plot: Employment Growth vs MW Gap at Federal Cutoff",
        "Bandwidth Sensitivity"
    ]
)

# Bin the running variable for cleaner visualization
rd_clean['mw_gap_bin'] = pd.cut(rd_clean['mw_gap'], bins=20)
binned = rd_clean.groupby('mw_gap_bin').agg(
    mw_gap_mid=('mw_gap','mean'),
    emp_growth_mean=('emp_growth','mean'),
    count=('emp_growth','count')
).reset_index()
binned = binned[binned['count'] >= 3]

# Scatter of binned means
left_bins  = binned[binned['mw_gap_mid'] <= 0]
right_bins = binned[binned['mw_gap_mid'] > 0]

fig2.add_trace(go.Scatter(
    x=left_bins['mw_gap_mid'], y=left_bins['emp_growth_mean'],
    mode='markers', name='At federal MW',
    marker=dict(color='#60a5fa', size=8)
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=right_bins['mw_gap_mid'], y=right_bins['emp_growth_mean'],
    mode='markers', name='Above federal MW',
    marker=dict(color='#34d399', size=8)
), row=1, col=2)

# Fitted lines
x_left  = np.linspace(-0.30, 0, 100)
x_right = np.linspace(0, 0.30, 100)
y_left  = left_model.params['Intercept']  + left_model.params['mw_gap']  * x_left
y_right = right_model.params['Intercept'] + right_model.params['mw_gap'] * x_right

fig2.add_trace(go.Scatter(
    x=x_left, y=y_left, mode='lines',
    line=dict(color='#60a5fa', width=2.5), showlegend=False
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=right_bins['mw_gap_mid'], y=right_bins['emp_growth_mean'],
    mode='markers', name='Above federal MW',
    marker=dict(color='#34d399', size=8), showlegend=False
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=x_right, y=y_right, mode='lines',
    line=dict(color='#34d399', width=2.5), showlegend=False
), row=1, col=1)

# Cutoff line
fig2.add_vline(x=0, line_dash="dash", line_color="#f59e0b",
               line_width=2, row=1, col=1)
fig2.add_annotation(
    x=0.02, y=binned['emp_growth_mean'].max()*0.8,
    text=f"RD = {rd_coef*100:.3f}pp",
    font=dict(color="#f59e0b", size=11),
    showarrow=False, row=1, col=1
)

# Bandwidth sensitivity plot
bws    = [0.10, 0.15, 0.20, 0.25, 0.30]
bw_coefs, bw_ses = [], []
for bw in bws:
    bw_data = rd_data[
        (~rd_data['year'].isin(fed_jump_years)) &
        (rd_data['emp_growth'].notna()) &
        (rd_data['mw_gap'].between(-bw, bw))
    ].copy()
    bw_data['above_x_gap'] = bw_data['above_cutoff'] * bw_data['mw_gap']
    try:
        m = smf.ols('emp_growth ~ above_cutoff + mw_gap + above_x_gap',
                    data=bw_data).fit(cov_type='HC3')
        bw_coefs.append(m.params['above_cutoff'] * 100)
        bw_ses.append(m.bse['above_cutoff'] * 100)
    except:
        bw_coefs.append(None); bw_ses.append(None)

fig2.add_trace(go.Scatter(
    x=[b*100 for b in bws], y=bw_coefs,
    mode='lines+markers', name='RD estimate',
    line=dict(color='#60a5fa', width=2),
    marker=dict(size=8),
    error_y=dict(type='data', array=[s*1.96 for s in bw_ses],
                 color='#374151', thickness=1.5),
    showlegend=False
), row=1, col=2)
fig2.add_hline(y=0, line_dash="dot", line_color="#6b7280",
               line_width=1, row=1, col=2)

fig2.update_layout(
    title="Method 2: Regression Discontinuity — MW Gap at Federal Cutoff",
    template="plotly_dark", height=480,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color='white')
)
fig2.update_xaxes(gridcolor='#1f2937',
                  title_text="MW Gap (% above federal)",    row=1, col=1)
fig2.update_xaxes(gridcolor='#1f2937',
                  title_text="Bandwidth (% of federal MW)", row=1, col=2)
fig2.update_yaxes(gridcolor='#1f2937',
                  title_text="Employment Growth (log)",     row=1, col=1)
fig2.update_yaxes(gridcolor='#1f2937',
                  title_text="RD Estimate (pp)",            row=1, col=2)
fig2.show()

print("\n" + "=" * 60)
print("RD COMPLETE")
print("=" * 60)
