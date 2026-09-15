"""Wane demo — five vapers, two engines, the curves draw themselves forward.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from profiles import PROFILES, START_MG
from engine import FixedTaper, AdaptiveTaper
from simulate import run_one, run_many, summarise

PINE, TEAL, ALARM, MUTE, CREAM = "#0E2B26", "#4FBFA8", "#E08163", "#8FA7A2", "#F1F6F4"

st.set_page_config(page_title="Wane — adaptive nicotine tapering", layout="wide")
st.markdown(f"""<style>
.stApp {{ background:{PINE}; color:{CREAM}; }}
h1,h2,h3,p,label,.stMarkdown {{ color:{CREAM} !important; }}
div[data-testid="stMetricValue"] {{ color:{TEAL}; }}
</style>""", unsafe_allow_html=True)

st.title("Wane — adaptive nicotine tapering")
st.caption("Five simulated vapers · the same people under a traditional fixed taper and under the Wane engine · synthetic behavioural model, parameters from published puff-topography and withdrawal studies")

# ---------------- controls ----------------
c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
weekly_cut = c1.slider("Weekly cut (both engines start here)", 0.05, 0.25, 0.12, 0.01, format="%d%%" if False else "%.2f")
weeks = c2.slider("Weeks", 8, 20, 14)
n_runs = c3.select_slider("Runs per profile (noise)", [1, 10, 30, 100], value=30)
week_shown = c4.slider("▶ Show up to week", 1, weeks, weeks)


@st.cache_data(show_spinner="Simulating…")
def cached_runs(n_runs, weeks, weekly_cut):
    return run_many(n_runs=n_runs, weeks=weeks, weekly_cut=weekly_cut)


runs = cached_runs(n_runs, weeks, weekly_cut)
runs = runs[runs["week"] <= week_shown]

# ---------------- headline numbers ----------------
summ = summarise(runs)
fixed = summ[summ.engine.str.startswith("Fixed")]
adapt = summ[summ.engine.str.startswith("Wane")]
m1, m2, m3 = st.columns(3)
m1.metric("Relapse rate — fixed taper", f"{fixed.relapse_rate.mean():.0%}")
m2.metric("Relapse rate — Wane", f"{adapt.relapse_rate.mean():.0%}",
          delta=f"{(adapt.relapse_rate.mean() - fixed.relapse_rate.mean()):+.0%}", delta_color="inverse")
m3.metric("Profiles still on the curve at week " + str(week_shown),
          f"{(adapt.relapse_rate < 0.5).sum()} / 5 vs {(fixed.relapse_rate < 0.5).sum()} / 5")


# ---------------- the two panels ----------------
def dose_panel(engine_prefix, title):
    fig = go.Figure()
    sub = runs[runs.engine.str.startswith(engine_prefix)]
    for p in PROFILES:
        d = sub[sub.profile == p.name].groupby("week")["dose"]
        med, lo, hi = d.median(), d.quantile(0.1), d.quantile(0.9)
        fig.add_trace(go.Scatter(x=list(hi.index) + list(lo.index[::-1]), y=list(hi) + list(lo[::-1]),
                                 fill="toself", fillcolor="rgba(79,191,168,0.08)", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=med.index, y=med.values, name=p.name, mode="lines+markers",
                                 line=dict(width=2.5)))
    fig.update_layout(title=title, paper_bgcolor=PINE, plot_bgcolor="#164038", font_color=CREAM,
                      xaxis=dict(title="week", range=[0.5, weeks + 0.5], gridcolor="#2C534B"),
                      yaxis=dict(title="nicotine mg/ml", range=[0, START_MG + 1], gridcolor="#2C534B"),
                      height=420, legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=60))
    return fig


left, right = st.columns(2)
left.plotly_chart(dose_panel("Fixed", "Traditional: cut 12 % every week, whatever happens"), use_container_width=True)
right.plotly_chart(dose_panel("Wane", "Wane: the same people, cut decided from their measured puffs"), use_container_width=True)
st.caption("Line = median of the runs · band = 10th–90th percentile · a line that jumps back to 20 mg/ml is a relapse")

# ---------------- per-profile table ----------------
tbl = summ.pivot(index="profile", columns="engine", values=["relapse_rate", "final_dose_median"])
tbl.columns = [f"{a} — {b.split(' ')[0]}" for a, b in tbl.columns]
st.dataframe(tbl.style.format({c: "{:.0%}" for c in tbl.columns if "relapse" in c} | {c: "{:.1f} mg/ml" for c in tbl.columns if "dose" in c}),
             use_container_width=True)

# ---------------- zoom on one person ----------------
st.subheader("Inside one person")
zc1, zc2 = st.columns([1, 3])
who = zc1.selectbox("Profile", [p.name for p in PROFILES], index=2)
seed = zc1.number_input("Run (seed)", 0, 999, 0)
prof = next(p for p in PROFILES if p.name == who)
zc1.markdown(f"*{prof.story}*")
zc1.markdown(f"elasticity **{prof.elasticity}** · craving sensitivity **{prof.craving_sensitivity}** · relapse threshold **{prof.relapse_threshold}**")

df, log = run_one(prof, AdaptiveTaper, weeks, seed=int(seed), weekly_cut=weekly_cut)
df = df[df.week <= week_shown]
fig = go.Figure()
fig.add_trace(go.Bar(x=df.day, y=df.puffs, name="puffs/day (measured)", marker_color="#2C534B"))
fig.add_trace(go.Scatter(x=df.day, y=df.dose * (df.puffs.max() / START_MG), name="dose (scaled)", line=dict(color=TEAL, width=3)))
fig.add_trace(go.Scatter(x=df.day, y=df.craving * (df.puffs.max() / 10), name="craving (model truth, scaled)", line=dict(color=ALARM, width=2, dash="dot")))
if log:
    for i, e in enumerate(log[: week_shown]):
        if e["action"] in ("HOLD", "HALF CUT"):
            fig.add_vline(x=7 * (i + 1), line=dict(color=ALARM if e["action"] == "HOLD" else MUTE, width=1, dash="dash"))
fig.update_layout(paper_bgcolor=PINE, plot_bgcolor="#164038", font_color=CREAM, height=380,
                  xaxis=dict(title="day", gridcolor="#2C534B"), yaxis=dict(title="puffs / day", gridcolor="#2C534B"),
                  legend=dict(orientation="h", y=-0.25), margin=dict(t=20, b=60))
zc2.plotly_chart(fig, use_container_width=True)
zc2.caption("Dashed lines: weeks where the engine held (orange) or halved the cut (grey) because the measured puffs said so.")

if log:
    lg = pd.DataFrame(log[: week_shown])
    lg.index = [f"week {i+1} → {i+2}" for i in range(len(lg))]
    st.dataframe(lg[["action", "cut", "rate", "risk", "crave", "puff_trend", "night_share", "ttfc_min"]]
                 .rename(columns={"rate": "personal rate (learned)", "crave": "derived craving", "puff_trend": "puff trend", "night_share": "night share", "ttfc_min": "time to first puff (min)"})
                 .style.format({"cut": "{:.0%}", "personal rate (learned)": "{:.0%}", "risk": "{:.2f}", "derived craving": "{:.1f}", "puff trend": "{:+.0%}", "night share": "{:.3f}", "time to first puff (min)": "{:.0f}"}),
                 use_container_width=True)
    st.caption("Every row is a decision the engine made and the measured features it made it from. No self-report in any column.")
