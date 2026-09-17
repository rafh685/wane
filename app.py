"""Wane demo: five vapers, two engines, the curves draw themselves forward.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from profiles import PROFILES, START_MG
from engine import FixedTaper, AdaptiveTaper, FittedTaper
from simulate import run_one, run_many, summarise

PINE, TEAL, ALARM, MUTE, CREAM = "#0E2B26", "#4FBFA8", "#E08163", "#8FA7A2", "#F1F6F4"

st.set_page_config(page_title="Wane: adaptive nicotine tapering", layout="wide")
st.markdown(f"""<style>
.stApp {{ background:{PINE}; color:{CREAM}; }}
h1,h2,h3,p,label,.stMarkdown {{ color:{CREAM} !important; }}
div[data-testid="stMetricValue"] {{ color:{TEAL}; }}
</style>""", unsafe_allow_html=True)

st.title("Wane: adaptive nicotine tapering")
st.caption("Five simulated vapers · the same people under a traditional fixed taper and under the Wane engine · synthetic behavioural model, parameters from published puff-topography and withdrawal studies")

# ---------------- controls ----------------
import time

PERIODS = {7: "7 days (weekly refill)", 3: "3 days (pre-mixed kit)", 2: "2 days (pre-mixed kit)", 1: "1 day (metering pod)"}
c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
weekly_cut = c1.slider("Weekly cut (both engines start here)", 0.05, 0.25, 0.12, 0.01, format="%.2f")
period = c2.select_slider("Engine decides every", options=[7, 3, 2, 1], value=7, format_func=lambda d: PERIODS[d])
weeks = c3.slider("Weeks", 8, 40, 26)
n_runs = c4.select_slider("Runs per profile (noise)", [1, 10, 30, 100], value=30)
ENGINES = {"fitted weights (learned on a synthetic population)": FittedTaper, "hand-set weights": AdaptiveTaper}
DELIVERY = {"manual: the user mixes and switches liquid at every step (adherence 0.85)": 0.85,
            "manual, less diligent user (adherence 0.70)": 0.70,
            "automatic: the device applies every step (hardware)": None}
r1, r2 = st.columns([1, 1])
engine_label = r1.radio("Wane engine", list(ENGINES), horizontal=False)
delivery_label = r2.radio("How steps get applied (both engines)", list(DELIVERY), horizontal=False)
WaneEngine = ENGINES[engine_label]
adherence = DELIVERY[delivery_label]

st.session_state.setdefault("playhead", 1)     # where the animation is
st.session_state.setdefault("playing", False)

p1, p2, p3 = st.columns([1, 1, 6])
if p1.button("▶ Play" if not st.session_state.playing else "⏸ Pause", width="stretch"):
    st.session_state.playing = not st.session_state.playing
    if st.session_state.playing and st.session_state.playhead >= weeks:
        st.session_state.playhead = 1
if p2.button("↺ Reset", width="stretch"):
    st.session_state.playing = False
    st.session_state.playhead = 1

# the slider FOLLOWS the playhead while playing; dragging it manually sets the playhead
week_shown = p3.slider("Week", 1, weeks, value=min(st.session_state.playhead, weeks), key=f"week_slider_{st.session_state.playhead}" if st.session_state.playing else "week_slider")
if not st.session_state.playing:
    st.session_state.playhead = week_shown


@st.cache_data(show_spinner="Simulating…")
def cached_runs(n_runs, weeks, weekly_cut, period, engine_label, delivery_label):
    return run_many(n_runs=n_runs, weeks=weeks, weekly_cut=weekly_cut, period=period,
                    engines=(FixedTaper, ENGINES[engine_label]), adherence=DELIVERY[delivery_label])


runs = cached_runs(n_runs, weeks, weekly_cut, period, engine_label, delivery_label)
runs = runs[runs["week"] <= week_shown]

# ---------------- headline numbers ----------------
summ = summarise(runs)
fixed = summ[summ.engine.str.startswith("Fixed")]
adapt = summ[summ.engine.str.startswith("Wane")]
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Relapse, fixed taper", f"{fixed.relapse_rate.mean():.0%}")
m2.metric("Relapse, Wane", f"{adapt.relapse_rate.mean():.0%}",
          delta=f"{(adapt.relapse_rate.mean() - fixed.relapse_rate.mean()):+.0%}", delta_color="inverse")
m3.metric("Stalled on the ladder, fixed", f"{fixed.stall_rate.mean():.0%}")
m4.metric("Stalled, Wane", f"{adapt.stall_rate.mean():.0%}",
          delta=f"{(adapt.stall_rate.mean() - fixed.stall_rate.mean()):+.0%}", delta_color="inverse")
m5.metric("Off nicotine, fixed", f"{fixed.off_rate.mean():.0%}")
m6.metric("Off nicotine, Wane", f"{adapt.off_rate.mean():.0%}",
          delta=f"{(adapt.off_rate.mean() - fixed.off_rate.mean()):+.0%}")
if adherence is not None:
    st.caption("Manual delivery: at each step the user has to mix or buy a weaker liquid and switch. The chance they do falls when craving is up and drifts down over the weeks. "
               "Stalled = eight scheduled steps in a row not applied. Automatic delivery removes this entirely: that is what the hardware is for.")


# ---------------- the two panels ----------------
COLORS = {"Marta": "#4FBFA8", "Diego": "#8FB8E8", "Lucía": "#E08163", "Karim": "#E8C46A", "Ana": "#C9A0E8"}
show_runs = st.toggle("Show every individual run (✕ = relapse)", value=False)


def dose_panel(engine_prefix, title):
    fig = go.Figure()
    sub = runs[runs.engine.str.startswith(engine_prefix)]
    for p in PROFILES:
        pp = sub[sub.profile == p.name]
        col = COLORS[p.name]
        if show_runs:
            # every run as a faint line; the first day of relapse as a red cross
            for s_, r in pp.groupby("seed"):
                wk = r.groupby("week")["dose"].first()
                fig.add_trace(go.Scatter(x=wk.index, y=wk.values, mode="lines", line=dict(color=col, width=1),
                                         opacity=0.25, showlegend=False, hoverinfo="skip"))
                rel = r[r.relapsed]
                if len(rel):
                    w0 = int(rel.week.min())
                    fig.add_trace(go.Scatter(x=[w0], y=[START_MG], mode="markers",
                                             marker=dict(symbol="x", size=10, color="#FF5A4A"),
                                             showlegend=False, hoverinfo="skip"))
        d = pp.groupby("week")["dose"]
        med, lo, hi = d.median(), d.quantile(0.1), d.quantile(0.9)
        if not show_runs:
            fig.add_trace(go.Scatter(x=list(hi.index) + list(lo.index[::-1]), y=list(hi) + list(lo[::-1]),
                                     fill="toself", fillcolor="rgba(79,191,168,0.08)", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=med.index, y=med.values, name=p.name, mode="lines+markers",
                                 line=dict(width=2.5, color=col)))
    fig.update_layout(title=title, paper_bgcolor=PINE, plot_bgcolor="#164038", font_color=CREAM,
                      xaxis=dict(title="week", range=[0.5, weeks + 0.5], gridcolor="#2C534B"),
                      yaxis=dict(title="nicotine mg/ml", range=[-0.5, START_MG + 1], gridcolor="#2C534B"),
                      height=420, legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=60))
    return fig


left, right = st.columns(2)
left.plotly_chart(dose_panel("Fixed", f"Traditional: {weekly_cut:.0%} per week in steps every {period} day(s), whatever happens"), width="stretch")
right.plotly_chart(dose_panel("Wane", "Wane: the same people, cut decided from their measured puffs"), width="stretch")
st.caption("Line = median of the runs · band = 10th–90th percentile · ✕ at 20 mg/ml = the week that run relapsed (back to disposables)")

# ---------------- relapse bars ----------------
rb = summ.copy()
rb["engine"] = rb.engine.str.split(" ").str[0]
figb = go.Figure()
for eng, col in (("Fixed", "#8FA7A2"), ("Wane", TEAL)):
    e = rb[rb.engine == eng].set_index("profile").reindex([p.name for p in PROFILES])
    figb.add_trace(go.Bar(x=e.index, y=e.relapse_rate, name=eng, marker_color=col,
                          text=[f"{v:.0%}" for v in e.relapse_rate], textposition="outside"))
figb.update_layout(barmode="group", paper_bgcolor=PINE, plot_bgcolor="#164038", font_color=CREAM, height=340,
                   yaxis=dict(title="relapse rate", tickformat=".0%", range=[0, 1.15], gridcolor="#2C534B"),
                   legend=dict(orientation="h", x=1, xanchor="right", y=0.98, yanchor="top", bgcolor="rgba(0,0,0,0)"),
                   margin=dict(t=60, b=30), title=dict(text="Who the ladder breaks, and who Wane keeps", y=0.95))
st.plotly_chart(figb, width="stretch")

# ---------------- per-profile table ----------------
tbl = summ.pivot(index="profile", columns="engine", values=["relapse_rate", "stall_rate", "off_rate", "weeks_to_zero_median"])
tbl.columns = [f"{a} ({b.split(' ')[0]})" for a, b in tbl.columns]
st.dataframe(tbl.style.format({c: "{:.0%}" for c in tbl.columns if "rate" in c} | {c: "{:.0f} wk" for c in tbl.columns if "weeks" in c}, na_rep="not yet"),
             width="stretch")
st.caption("Off nicotine = reached 0 mg/ml and stayed there two clean weeks. Weeks to zero = median over the runs that got there.")

# ---------------- zoom on one person ----------------
st.subheader("Inside one person")
zc1, zc2 = st.columns([1, 3])
who = zc1.selectbox("Profile", [p.name for p in PROFILES], index=2)
seed = zc1.number_input("Run (seed)", 0, 999, 0)
prof = next(p for p in PROFILES if p.name == who)
zc1.markdown(f"*{prof.story}*")
zc1.markdown(f"elasticity **{prof.elasticity}** · craving sensitivity **{prof.craving_sensitivity}** · relapse threshold **{prof.relapse_threshold}**")

df, log = run_one(prof, WaneEngine, weeks, seed=int(seed), weekly_cut=weekly_cut, period=period, adherence=adherence)
log = [e for e in (log or []) if e["day"] <= 7 * week_shown]
df = df[df.week <= week_shown]
fig = go.Figure()
fig.add_trace(go.Bar(x=df.day, y=df.puffs, name="puffs/day (measured)", marker_color="#2C534B"))
fig.add_trace(go.Scatter(x=df.day, y=df.dose * (df.puffs.max() / START_MG), name="dose (scaled)", line=dict(color=TEAL, width=3)))
fig.add_trace(go.Scatter(x=df.day, y=df.craving * (df.puffs.max() / 10), name="craving (model truth, scaled)", line=dict(color=ALARM, width=2, dash="dot")))
for e in log:
    if e["action"] in ("HOLD", "HALF CUT"):
        fig.add_vline(x=e["day"], line=dict(color=ALARM if e["action"] == "HOLD" else MUTE, width=1, dash="dash"))
fig.update_layout(paper_bgcolor=PINE, plot_bgcolor="#164038", font_color=CREAM, height=380,
                  xaxis=dict(title="day", gridcolor="#2C534B"), yaxis=dict(title="puffs / day", gridcolor="#2C534B"),
                  legend=dict(orientation="h", y=-0.25), margin=dict(t=20, b=60))
zc2.plotly_chart(fig, width="stretch")
zc2.caption("Dashed lines: weeks where the engine held (orange) or halved the cut (grey) because the measured puffs said so.")

if log:
    lg = pd.DataFrame(log)
    lg.index = [f"day {int(d)}" for d in lg["day"]]
    st.dataframe(lg[["action", "cut", "rate", "risk", "crave", "puff_trend", "night_share", "ttfc_min"]]
                 .rename(columns={"rate": "personal rate (learned)", "crave": "derived craving", "puff_trend": "puff trend", "night_share": "night share", "ttfc_min": "time to first puff (min)"})
                 .style.format({"cut": "{:.0%}", "personal rate (learned)": "{:.0%}", "risk": "{:.2f}", "derived craving": "{:.1f}", "puff trend": "{:+.0%}", "night share": "{:.3f}", "time to first puff (min)": "{:.0f}"}),
                 width="stretch")
    st.caption("Every row is a decision the engine made and the measured features it made it from. No self-report in any column. Personal rate is shown per week whatever the decision period.")


# ---------------- autoplay: advance one week per tick while playing ----------------
if st.session_state.playing:
    if st.session_state.playhead < weeks:
        time.sleep(0.7)
        st.session_state.playhead += 1
        st.rerun()
    else:
        st.session_state.playing = False
        st.rerun()
