from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


# ================================================================
# InsightFlow | Premium Streamlit dashboard for CDSP final project
# ================================================================
st.set_page_config(
    page_title="InsightFlow | Bank Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
MODEL_PATH = ROOT / "models" / "bank_marketing_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.pkl"

PALETTE = {
    "bg": "#0B1020",
    "surface": "#111A2E",
    "surface2": "#17233D",
    "text": "#ECF2FF",
    "muted": "#9BA9C7",
    "primary": "#57D8BE",
    "accent": "#7C6CFF",
    "warning": "#F8C76B",
    "danger": "#FF7B93",
    "grid": "rgba(151,169,207,.14)",
}


def inject_css():
    """Global Streamlit styling with custom CSS."""
    st.markdown(
        f"""
        <style>
        :root {{
            --bg:{PALETTE['bg']}; --surface:{PALETTE['surface']};
            --surface2:{PALETTE['surface2']}; --text:{PALETTE['text']};
            --muted:{PALETTE['muted']}; --primary:{PALETTE['primary']};
            --accent:{PALETTE['accent']}; --warning:{PALETTE['warning']};
            --danger:{PALETTE['danger']}; --grid:{PALETTE['grid']};
        }}
        .stApp {{
            background:
                radial-gradient(circle at 92% -10%, rgba(124,108,255,.18), transparent 31%),
                radial-gradient(circle at 0% 15%, rgba(87,216,190,.10), transparent 24%),
                var(--bg);
            color:var(--text);
            font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        #MainMenu, footer, header {{visibility:hidden;}}
        .block-container {{max-width:1440px; padding-top:1.7rem; padding-bottom:2.2rem;}}
        [data-testid="stSidebar"] {{background:linear-gradient(180deg,#0d1528,#0b1020); border-right:1px solid rgba(151,169,207,.18);}}
        [data-testid="stSidebar"] .stRadio label {{border-radius:11px; padding:.38rem .52rem; transition:.18s ease;}}
        [data-testid="stSidebar"] .stRadio label:hover {{background:rgba(124,108,255,.14);}}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stCaption {{color:var(--muted);}}
        h1,h2,h3,h4 {{color:var(--text)!important; letter-spacing:-.025em;}}
        .hero {{
            overflow:hidden; position:relative; padding:2rem 2.15rem; margin:0 0 1.15rem;
            border:1px solid rgba(151,169,207,.19); border-radius:24px;
            background:linear-gradient(110deg,rgba(17,26,46,.98),rgba(23,35,61,.85));
            box-shadow:0 22px 55px rgba(0,0,0,.24);
        }}
        .hero:after {{
            content:""; position:absolute; right:-75px; top:-94px; width:250px; height:250px;
            border-radius:50%; border:1px solid rgba(87,216,190,.33);
            box-shadow:0 0 0 24px rgba(87,216,190,.045),0 0 0 52px rgba(124,108,255,.04);
        }}
        .eyebrow {{color:var(--primary); font-size:.75rem; text-transform:uppercase; letter-spacing:.15em; font-weight:800; margin-bottom:.55rem;}}
        .hero h1 {{font-size:clamp(2rem,3.6vw,3.15rem); line-height:1.03; max-width:880px; margin:0;}}
        .hero p {{color:var(--muted); font-size:1rem; max-width:760px; line-height:1.62; margin:.8rem 0 0;}}
        .pills {{display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.2rem;}}
        .pill {{border:1px solid rgba(151,169,207,.18); background:rgba(255,255,255,.035); border-radius:999px; padding:.45rem .72rem; font-size:.78rem; color:#dce8ff; font-weight:700;}}
        .dot {{display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--primary); box-shadow:0 0 12px var(--primary); margin-right:.42rem;}}
        .section-head {{display:flex; justify-content:space-between; align-items:baseline; gap:1rem; margin:1.1rem 0 .42rem;}}
        .section-head h3 {{font-size:1.08rem; margin:0;}}
        .section-head span {{color:var(--muted); font-size:.8rem;}}
        .metric-card {{
            --tone:var(--primary); min-height:140px; position:relative; overflow:hidden;
            padding:1rem; border-radius:18px; border:1px solid rgba(151,169,207,.18);
            background:linear-gradient(135deg,rgba(23,35,61,.95),rgba(17,26,46,.95));
            transition:transform .18s ease,border-color .18s ease;
        }}
        .metric-card:hover {{transform:translateY(-3px); border-color:rgba(87,216,190,.45);}}
        .metric-card:before {{content:""; position:absolute; top:0; left:0; width:4px; height:100%; background:var(--tone);}}
        .metric-label {{color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.1em; font-weight:800;}}
        .metric-value {{color:var(--text); font-size:1.75rem; font-weight:900; letter-spacing:-.05em; margin-top:.35rem; white-space:pre-line;}}
        .metric-note {{color:var(--muted); font-size:.78rem; line-height:1.4; margin-top:.5rem;}}
        .insight-card {{min-height:122px; padding:1rem; border:1px solid rgba(151,169,207,.18); border-radius:16px; background:rgba(17,26,46,.80);}}
        .insight-index {{display:inline-flex; align-items:center; justify-content:center; width:29px; height:29px; border-radius:9px; background:rgba(124,108,255,.16); color:#cac3ff; font-size:.76rem; font-weight:850;}}
        .insight-title {{font-weight:800; color:var(--text); margin:.62rem 0 .24rem;}}
        .insight-body {{color:var(--muted); font-size:.81rem; line-height:1.45;}}
        .brand {{padding:.8rem 1rem .65rem; display:flex; align-items:center;}}
        .brand-mark {{width:38px; height:38px; display:grid; place-items:center; border-radius:12px; background:linear-gradient(135deg,var(--primary),var(--accent)); color:#07111f; font-weight:950; margin-right:.6rem; box-shadow:0 10px 22px rgba(87,216,190,.16);}}
        .brand-name {{font-weight:900; font-size:1.05rem; letter-spacing:-.03em;}}
        .brand-sub {{color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; margin-top:.14rem;}}
        .side-card {{margin:1rem .7rem .25rem; padding:.9rem; border-radius:16px; background:rgba(124,108,255,.10); border:1px solid rgba(124,108,255,.26);}}
        .side-card small {{color:#c5beff; text-transform:uppercase; letter-spacing:.1em; font-size:.68rem; font-weight:800;}}
        .side-card strong {{display:block; color:var(--text); margin-top:.35rem; font-size:.9rem;}}
        [data-baseweb="select"]>div,[data-baseweb="input"]>div {{background:rgba(7,13,27,.72)!important; border-color:rgba(151,169,207,.25)!important; border-radius:11px!important; color:var(--text)!important;}}
        [data-baseweb="input"] input {{color:var(--text)!important;}}
        [data-testid="stNumberInput"] button {{background:rgba(124,108,255,.12)!important; color:var(--text)!important;}}
        .stButton>button,[data-testid="stFormSubmitButton"]>button {{min-height:45px; border:0!important; border-radius:12px!important; background:linear-gradient(135deg,var(--primary),#65b9ea)!important; color:#07111f!important; font-weight:900!important; transition:transform .18s ease,filter .18s ease;}}
        .stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover {{transform:translateY(-2px); filter:brightness(1.06);}}
        .prediction-panel {{display:flex; align-items:center; gap:1.1rem; padding:1.25rem; border-radius:20px; border:1px solid rgba(87,216,190,.30); background:linear-gradient(135deg,rgba(87,216,190,.13),rgba(124,108,255,.13)); margin-top:1rem;}}
        .gauge {{--score:50; width:118px; height:118px; flex:0 0 118px; display:grid; place-items:center; border-radius:50%; background:conic-gradient(var(--primary) calc(var(--score) * 1%),#26344f 0); position:relative;}}
        .gauge:before {{content:""; position:absolute; width:92px; height:92px; background:#10182c; border:1px solid rgba(255,255,255,.08); border-radius:50%;}}
        .gauge span {{z-index:1; position:relative; color:var(--text); font-size:1.25rem; font-weight:900;}}
        .badge {{display:inline-block; padding:.38rem .65rem; border-radius:999px; margin-bottom:.42rem; font-size:.75rem; font-weight:850;}}
        .method {{border-left:3px solid var(--accent); padding:.72rem .9rem; margin:.52rem 0; border-radius:0 10px 10px 0; background:rgba(124,108,255,.08); color:var(--muted); font-size:.86rem; line-height:1.5;}}
        .form-shell {{border:1px solid rgba(151,169,207,.18); border-radius:18px; padding:1.1rem 1.15rem .3rem; background:rgba(17,26,46,.78);}}
        .footer {{padding:1.8rem 0 .2rem; color:var(--muted); text-align:center; font-size:.76rem;}}
        @media(max-width:700px) {{.hero{{padding:1.4rem; border-radius:18px}} .prediction-panel{{flex-direction:column; align-items:flex-start;}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, sep=";")
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.strip()
    data["y_num"] = data["y"].map({"no": 0, "yes": 1})
    return data


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)


def stylize_chart(fig, height=365):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter, Arial, sans-serif"),
        margin=dict(l=12, r=12, t=54, b=16),
        title=dict(font=dict(size=16)),
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=PALETTE["muted"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False, color=PALETTE["muted"])
    return fig


def metric_card(label, value, note, tone):
    st.markdown(
        f'''<div class="metric-card" style="--tone:{tone}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def insight_card(number, title, detail):
    st.markdown(
        f'''<div class="insight-card">
        <div class="insight-index">{number:02d}</div>
        <div class="insight-title">{title}</div>
        <div class="insight-body">{detail}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def section_head(title, note=""):
    st.markdown(
        f'<div class="section-head"><h3>{title}</h3><span>{note}</span></div>',
        unsafe_allow_html=True,
    )


def pretty_label(name):
    return name.replace(".", " · ").replace("_", " ").title()


def render_js_signal():
    """Small decorative HTML/CSS/JS component in an isolated iframe."""
    components.html(
        """
        <!doctype html><html><head><style>
        html,body{margin:0;background:transparent;font-family:Arial,sans-serif;overflow:hidden}
        .signal{height:68px;box-sizing:border-box;border:1px solid rgba(151,169,207,.18);border-radius:16px;background:linear-gradient(90deg,rgba(17,26,46,.88),rgba(23,35,61,.74));display:flex;align-items:center;padding:0 18px;gap:12px;color:#9BA9C7;font-size:12px;letter-spacing:.08em;text-transform:uppercase}
        .dot{width:9px;height:9px;border-radius:50%;background:#57D8BE;box-shadow:0 0 0 0 rgba(87,216,190,.6);animation:pulse 1.7s infinite}
        @keyframes pulse{70%{box-shadow:0 0 0 10px rgba(87,216,190,0)}100%{box-shadow:0 0 0 0 rgba(87,216,190,0)}}
        canvas{flex:1;height:26px}.state{font-weight:700;color:#DCE8FF;white-space:nowrap}
        </style></head><body><div class="signal"><span class="dot"></span><span class="state">Model signal active</span><canvas id="wave"></canvas><span id="status">Decision ready</span></div>
        <script>
        const c=document.getElementById('wave'),ctx=c.getContext('2d');let t=0;
        function resize(){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0)}
        function draw(){const w=c.clientWidth,h=c.clientHeight;ctx.clearRect(0,0,w,h);ctx.beginPath();for(let x=0;x<w;x+=2){let y=h/2+Math.sin((x+t)*.05)*4+Math.sin((x+t)*.14)*2;x===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.strokeStyle='#57D8BE';ctx.lineWidth=1.5;ctx.shadowBlur=10;ctx.shadowColor='#57D8BE';ctx.stroke();ctx.shadowBlur=0;t+=1.6;requestAnimationFrame(draw)}
        window.addEventListener('resize',resize);resize();draw();
        const labels=['Decision ready','Signals synchronized','Model online'];setInterval(()=>document.getElementById('status').textContent=labels[Math.floor(Math.random()*labels.length)],2500);
        </script></body></html>
        """,
        height=72,
        scrolling=False,
    )


def feature_importance(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return None
    return pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False)


def prediction_card(probability, threshold):
    if probability >= 0.65:
        label, message, bg, fg = "High Priority Lead", "Strong candidate for focused outreach.", "#0D493F", "#9BF4DF"
    elif probability >= threshold:
        label, message, bg, fg = "Medium Priority Lead", "Worth contacting with a targeted campaign message.", "#453A0D", "#F8D978"
    else:
        label, message, bg, fg = "Low Priority Lead", "Lower immediate priority; consider a lighter-touch approach.", "#472330", "#FFB5C4"
    score = max(0, min(100, probability * 100))
    st.markdown(
        f'''<div class="prediction-panel">
            <div class="gauge" style="--score:{score:.2f}"><span>{probability:.0%}</span></div>
            <div><div class="badge" style="background:{bg};color:{fg}">{label}</div>
            <h3 style="margin:0 0 .35rem">Subscription likelihood estimate</h3>
            <div style="color:#9BA9C7;font-size:.9rem;line-height:1.55">{message}<br/>Operating threshold: <strong style="color:#ECF2FF">{threshold:.0%}</strong></div></div>
        </div>''',
        unsafe_allow_html=True,
    )


inject_css()

if not DATA_PATH.exists():
    st.error("Dataset missing. Put `bank-additional-full.csv` inside the `data` folder.")
    st.stop()
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model artifacts missing. Run the notebook cell that saves the model and metadata inside `models`.")
    st.stop()

df = load_data()
model, metadata = load_artifacts()
features = metadata["features"]
threshold = float(metadata["threshold"])
metrics = metadata["test_metrics"]
model_name = metadata["model_name"]

# Sidebar
st.sidebar.markdown('''<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">InsightFlow</div><div class="brand-sub">Bank Intelligence</div></div></div>''', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Command Center", "Data Explorer", "Prediction Studio", "Model Card"], label_visibility="collapsed")
st.sidebar.markdown(f'''<div class="side-card"><small>Active model</small><strong>{model_name}</strong><div style="color:#9BA9C7;margin-top:.45rem;font-size:.78rem">Test ROC-AUC: {metrics['ROC-AUC']:.3f}</div></div>''', unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption("CDSP Final Project · Bank Marketing Classification\n\nThe deployed model intentionally excludes `duration` to prevent data leakage.")

positive_rate = df["y_num"].mean()
job_rates = df.groupby("job", as_index=False)["y_num"].mean().sort_values("y_num", ascending=False)
contact_rates = df.groupby("contact", as_index=False)["y_num"].mean().sort_values("y_num", ascending=False)
outcome_rates = df.groupby("poutcome", as_index=False)["y_num"].mean().sort_values("y_num", ascending=False)

# Page 1: Command Center
if page == "Command Center":
    st.markdown('''<section class="hero"><div class="eyebrow">Machine Learning · Decision Intelligence</div><h1>Turn campaign data into <span style="color:#57D8BE">confident actions.</span></h1><p>An interactive decision-support dashboard that estimates a customer's probability of term-deposit subscription and helps teams prioritize their outreach intelligently.</p><div class="pills"><span class="pill"><span class="dot"></span> Production-minded model</span><span class="pill">◈ Leakage-aware feature set</span><span class="pill">↗ Interactive decision support</span></div></section>''', unsafe_allow_html=True)
    render_js_signal()

    section_head("Decision snapshot", "Model, data and campaign context")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Customer records", f"{len(df):,}", "Historical campaign observations", PALETTE["primary"])
    with c2: metric_card("Subscription rate", f"{positive_rate:.1%}", "Positive class in the target variable", PALETTE["accent"])
    with c3: metric_card("Best model", model_name.replace(" ", "\n"), "Selected using cross-validation", PALETTE["warning"])
    with c4: metric_card("Test ROC-AUC", f"{metrics['ROC-AUC']:.3f}", "Ranking quality on unseen data", PALETTE["danger"])

    section_head("What the data is signaling", "Observed associations — not causal claims")
    i1, i2, i3 = st.columns(3)
    with i1: insight_card(1, "Strongest job segment", f"{job_rates.iloc[0]['job']} has the highest observed subscription rate at {job_rates.iloc[0]['y_num']:.1%}.")
    with i2: insight_card(2, "Best contact channel", f"{contact_rates.iloc[0]['contact']} has the highest observed subscription rate at {contact_rates.iloc[0]['y_num']:.1%}.")
    with i3: insight_card(3, "Previous campaign signal", f"{outcome_rates.iloc[0]['poutcome']} records have the highest observed subscription rate at {outcome_rates.iloc[0]['y_num']:.1%}.")

    left, right = st.columns([1, 1.12])
    with left:
        target = df["y"].value_counts().rename_axis("Subscription").reset_index(name="Customers")
        fig = px.pie(target, names="Subscription", values="Customers", hole=.68, color="Subscription", color_discrete_map={"yes": PALETTE["primary"], "no": "#31425F"})
        fig.update_traces(textinfo="percent", textfont=dict(color=PALETTE["text"], size=14), hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>")
        fig = stylize_chart(fig, 350)
        fig.update_layout(title="Subscription distribution", annotations=[dict(text="Target<br>balance", x=.5, y=.5, showarrow=False, font=dict(size=16, color=PALETTE["text"]))])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        view = contact_rates.copy(); view["Rate"] = view["y_num"] * 100
        fig = px.bar(view, x="contact", y="Rate", text="Rate", color="contact", color_discrete_sequence=[PALETTE["accent"], PALETTE["primary"]])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = stylize_chart(fig, 350); fig.update_layout(title="Subscription rate by contact channel", showlegend=False)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('''<div class="method"><strong style="color:#ECF2FF">Production design note:</strong> <code>duration</code> is intentionally excluded from the deployed model. It becomes known only after a phone call is completed, so including it in pre-call targeting would create data leakage.</div>''', unsafe_allow_html=True)

# Page 2: Explorer
elif page == "Data Explorer":
    st.markdown('''<div class="eyebrow">Interactive analysis</div><h1 style="margin-bottom:.25rem">Explore campaign patterns</h1><p style="color:#9BA9C7;margin-top:0">Filter the historical data to inspect associations between customer characteristics, campaign behavior and subscription outcomes.</p>''', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1.05, 1.05, 1.25])
    with f1: selected_jobs = st.multiselect("Job segment", sorted(df["job"].dropna().unique()), placeholder="All job segments")
    with f2: selected_months = st.multiselect("Campaign month", sorted(df["month"].dropna().unique()), placeholder="All months")
    with f3:
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))

    filtered = df.copy()
    if selected_jobs: filtered = filtered[filtered["job"].isin(selected_jobs)]
    if selected_months: filtered = filtered[filtered["month"].isin(selected_months)]
    filtered = filtered[filtered["age"].between(age_range[0], age_range[1])]
    if filtered.empty:
        st.warning("No records match the selected filters. Expand the selection and try again.")
        st.stop()

    filtered_rate = filtered["y_num"].mean()
    section_head("Filtered cohort", f"{len(filtered):,} records · {filtered_rate:.1%} subscription rate")
    a, b, c = st.columns(3)
    with a: metric_card("Filtered records", f"{len(filtered):,}", "Current explorer selection", PALETTE["primary"])
    with b: metric_card("Positive rate", f"{filtered_rate:.1%}", "Within selected cohort", PALETTE["accent"])
    with c: metric_card("Age span", f"{age_range[0]}–{age_range[1]}", "Applied customer filter", PALETTE["warning"])

    l1, r1 = st.columns(2)
    with l1:
        view = filtered.groupby("job", as_index=False)["y_num"].mean().sort_values("y_num")
        view["Rate"] = view["y_num"] * 100
        fig = px.bar(view, x="Rate", y="job", orientation="h", color="Rate", color_continuous_scale=[[0,"#31425F"],[.55,PALETTE["accent"]],[1,PALETTE["primary"]]])
        fig = stylize_chart(fig, 420); fig.update_layout(title="Subscription rate by job", coloraxis_showscale=False); fig.update_xaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with r1:
        order = ["mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        view = filtered.groupby("month", as_index=False)["y_num"].mean(); view["Rate"] = view["y_num"] * 100
        view["month"] = pd.Categorical(view["month"], categories=order, ordered=True); view = view.sort_values("month")
        fig = px.area(view, x="month", y="Rate", markers=True)
        fig.update_traces(line=dict(color=PALETTE["primary"], width=3), fillcolor="rgba(87,216,190,.16)", marker=dict(size=7, color=PALETTE["primary"]))
        fig = stylize_chart(fig, 420); fig.update_layout(title="Subscription rate by campaign month"); fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    l2, r2 = st.columns(2)
    with l2:
        view = filtered.groupby("poutcome", as_index=False)["y_num"].mean().sort_values("y_num", ascending=False); view["Rate"] = view["y_num"] * 100
        fig = px.bar(view, x="poutcome", y="Rate", text="Rate", color="poutcome", color_discrete_sequence=[PALETTE["primary"], PALETTE["accent"], "#476281"])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = stylize_chart(fig, 355); fig.update_layout(title="Subscription rate by previous campaign outcome", showlegend=False); fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with r2:
        view = filtered.groupby("campaign", as_index=False)["y_num"].mean(); view = view[view["campaign"] <= 10]; view["Rate"] = view["y_num"] * 100
        fig = px.line(view, x="campaign", y="Rate", markers=True)
        fig.update_traces(line=dict(color=PALETTE["warning"], width=3), marker=dict(size=7, color=PALETTE["warning"]))
        fig = stylize_chart(fig, 355); fig.update_layout(title="Subscription rate by number of contacts"); fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("View filtered records"):
        st.dataframe(filtered.drop(columns=["y_num"]).head(500), use_container_width=True, hide_index=True)

# Page 3: Prediction Studio
elif page == "Prediction Studio":
    st.markdown('''<div class="eyebrow">Decision support</div><h1 style="margin-bottom:.25rem">Prediction Studio</h1><p style="color:#9BA9C7;margin-top:0">Enter a prospect profile. The model returns a probability estimate and a practical outreach priority.</p>''', unsafe_allow_html=True)
    customer = [x for x in ["age","job","marital","education","default","housing","loan"] if x in features]
    campaign = [x for x in ["contact","month","day_of_week","campaign","pdays","previous","poutcome"] if x in features]
    economic = [x for x in ["emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"] if x in features]
    values = {}

    def input_widget(feature, container):
        with container:
            series = df[feature]
            if series.dtype == "object":
                options = sorted(series.dropna().unique())
                default = series.mode().iloc[0]
                values[feature] = st.selectbox(pretty_label(feature), options, index=options.index(default))
            elif pd.api.types.is_integer_dtype(series):
                values[feature] = st.number_input(pretty_label(feature), min_value=int(series.min()), max_value=int(series.max()), value=int(series.median()), step=1)
            else:
                values[feature] = st.number_input(pretty_label(feature), value=float(series.median()), step=0.01, format="%.3f")

    st.markdown('<div class="form-shell">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown("#### Customer profile")
        cols = st.columns(3)
        for i, feature in enumerate(customer): input_widget(feature, cols[i % 3])
        st.markdown("#### Campaign context")
        cols = st.columns(3)
        for i, feature in enumerate(campaign): input_widget(feature, cols[i % 3])
        st.markdown("#### Economic indicators")
        cols = st.columns(3)
        for i, feature in enumerate(economic): input_widget(feature, cols[i % 3])
        submitted = st.form_submit_button("Generate subscription estimate", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        input_df = pd.DataFrame([values], columns=features)
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction_card(probability, threshold)
        st.markdown('''<div class="method">The priority label is an operational recommendation, not a guarantee. The model estimates probability from historical patterns; the <strong style="color:#ECF2FF">65%</strong> high-priority band is a configurable business rule.</div>''', unsafe_allow_html=True)

# Page 4: Model card
else:
    st.markdown('''<div class="eyebrow">Model governance</div><h1 style="margin-bottom:.25rem">Model Card</h1><p style="color:#9BA9C7;margin-top:0">A concise record of what the model does, how it was evaluated, and where its limits begin.</p>''', unsafe_allow_html=True)
    section_head("Evaluation on unseen test data", "Results saved after model training")
    names = [("Accuracy",PALETTE["primary"]),("Precision",PALETTE["accent"]),("Recall",PALETTE["warning"]),("F1 Score",PALETTE["danger"]),("ROC-AUC","#8AA8FF")]
    cols = st.columns(5)
    for col, (name, tone) in zip(cols, names):
        with col: metric_card(name, f"{metrics[name]:.3f}", "Held-out test set", tone)

    left, right = st.columns([1.15, 1])
    with left:
        importance = feature_importance(model)
        if importance is not None:
            view = importance.head(15).sort_values("Importance")
            fig = px.bar(view, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=[[0,"#31425F"],[.55,PALETTE["accent"]],[1,PALETTE["primary"]]])
            fig = stylize_chart(fig, 490); fig.update_layout(title="Top predictive features", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Feature importance is unavailable for the selected model.")
    with right:
        st.markdown("### Methodology")
        st.markdown('''<div class="method"><strong style="color:#ECF2FF">Problem type:</strong><br/>Binary Classification — predict yes/no subscription.</div><div class="method"><strong style="color:#ECF2FF">Model selection:</strong><br/>Compare models using five-fold cross-validation, then test the selected model on untouched data.</div><div class="method"><strong style="color:#ECF2FF">Preprocessing:</strong><br/>Median imputation and scaling for numeric fields; most-frequent imputation and One-Hot Encoding for categories.</div><div class="method"><strong style="color:#ECF2FF">Leakage control:</strong><br/>Exclude <code>duration</code> because it becomes known only after the phone call.</div>''', unsafe_allow_html=True)
        st.markdown("### Interpretation boundary")
        st.caption("Feature importance reflects predictive association in this dataset. It does not establish a causal relationship.")
        st.markdown("### Deployment note")
        st.caption("The saved pipeline contains preprocessing and the model together, so prediction inputs are transformed consistently with training.")

st.markdown('<div class="footer">InsightFlow · Bank Marketing Classification · CDSP Final Project</div>', unsafe_allow_html=True)
