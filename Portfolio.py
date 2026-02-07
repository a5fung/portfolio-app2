import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Portfolio Command Center", layout="wide", page_icon="🚀")

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    /* 1. Main Background */
    .stApp { background-color: #F8F9FA !important; }
    
    /* 2. Sidebar -> White */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
    }
    
    /* 3. Global Text -> Dark Grey */
    h1, h2, h3, h4, h5, h6, p, li, div, span, label {
        color: #1F2937 !important;
    }
    
    /* 4. KPI CARDS */
    .kpi-container {
        background-color: #E3F2FD;
        border: 1px solid #BBDEFB;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .kpi-label {
        font-size: 14px;
        color: #1565C0 !important;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #0D47A1 !important;
        margin-top: 5px;
    }

    /* 5. TABS */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #6B7280 !important;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1565C0 !important;
        border-bottom: 3px solid #1565C0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 🔒 SECURITY ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Please enter your password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password incorrect", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- 3. HELPER FUNCTIONS ---
def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1F2937"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            showgrid=True, gridcolor="#E0E0E0", gridwidth=1,
            showline=True, linecolor="#E0E0E0",
            tickfont=dict(color="#1F2937"),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#E0E0E0", gridwidth=1,
            showline=False, zeroline=False,
            tickfont=dict(color="#1F2937"),
        ),
        yaxis2=dict(
            showgrid=False, showline=False, zeroline=False,
            tickfont=dict(color="#1F2937"),
            overlaying="y", side="right"
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#1F2937"), bgcolor="rgba(255,255,255,0.8)"
        )
    )
    return fig

# --- 4. DATA LOADING ---
@st.cache_data(ttl=60)
def load_data():
    try:
        url = st.secrets["public_sheet_url"]
        if "PASTE_YOUR" in url: return pd.DataFrame()
        return pd.read_csv(url)
    except: return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Data not loaded. Check Secrets.")
    st.stop()

# --- CLEANING ---
df.columns = df.columns.str.strip()
df = df[[c for c in df.columns if "Unnamed" not in c]]
df = df.dropna(subset=["Bucket"])

for col in ["Total Value", "Cash", "Margin Balance"]:
    if col in df.columns:
        s = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
        df[col] = pd.to_numeric(s.str.replace('-', '0'), errors='coerce').fillna(0)

if "YTD" in df.columns:
    s = df["YTD"].astype(str).str.replace('%', '', regex=False)
    df["YTD"] = pd.to_numeric(s, errors='coerce').fillna(0)
else: df["YTD"] = 0.0

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.sort_values("Date")

# --- 5. FILTERS ---
with st.sidebar:
    st.header("🔍 Filters")
    min_d = df["Date"].min()
    max_d = df["Date"].max()
    
    default_start = date(2026, 1, 1)
    if default_start < min_d.date(): default_start = min_d.date()
        
    dr = st.date_input("Date Range", [default_start, max_d.date()])

if len(dr) == 2:
    f_df = df[(df["Date"] >= pd.to_datetime(dr[0])) & (df["Date"] <= pd.to_datetime(dr[1]))]
else: f_df = df

latest = f_df[f_df["Date"] == f_df["Date"].max()]
acct_order = latest.groupby("Account")["Total Value"].sum().sort_values(ascending=False).index.tolist()
val = latest["Total Value"].sum()
cash = latest["Cash"].sum()
mrg = latest["Margin Balance"].sum()

# --- 6. DASHBOARD ---
st.title("🚀 Portfolio Command Center")

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Net Liquidity", f"${val:,.0f}")
with c2: kpi_card("Cash", f"${cash:,.0f}")
with c3: kpi_card("Margin", f"${mrg:,.0f}")
with c4: kpi_card("Active Accounts", f"{len(acct_order)}")

st.markdown("---")

t1, t2, t3, t4 = st.tabs(["📊 Executive Summary", "📈 Trends (YTD)", "🥧 Allocation", "⚠️ Risk Monitor"])

# TAB 1: SUMMARY
with t1:
    col_trend, col_sun = st.columns([2, 1])
    
    with col_trend:
        st.subheader("Portfolio Growth")
        tr = f_df.groupby("Date")["Total Value"].sum().reset_index()
        fig = px.area(tr, x="Date", y="Total Value")
        fig.update_traces(
            mode="lines+markers+text", 
            line_color="#1565C0", 
            fillcolor="rgba(21, 101, 192, 0.1)",
            marker=dict(size=6),
            text=tr["Total Value"],
            texttemplate='%{y:.2s}',
            textposition="top center",
            textfont=dict(size=12, color="#1565C0")
        )
        # STATIC PLOT (Mobile Friendly)
        st.plotly_chart(style_chart(fig), use_container_width=True, config={'staticPlot': True})

    with col_sun:
        st.subheader("Allocation Hierarchy")
        fig_sun = px.sunburst(latest, path=['Bucket', 'Account'], values='Total Value')
        fig_sun.update_traces(textinfo="label+percent entry")
        st.plotly_chart(style_chart(fig_sun), use_container_width=True, config={'staticPlot': True})

    # Global Risk
    st.subheader("Global Risk Monitor")
    d_tot = f_df.groupby("Date")["Total Value"].sum().reset_index()
    g_hist = df.groupby("Date")["Total Value"].sum().reset_index()
    g_hist["Peak"] = g_hist["Total Value"].cummax()
    d_tot = pd.merge(d_tot, g_hist[["Date", "Peak"]], on="Date", how="left")
    
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=d_tot["Date"], y=d_tot["Total Value"], name='Value', 
        line=dict(color='#1565C0', width=3),
        mode="lines+text", text=d_tot["Total Value"], texttemplate='%{y:.2s}', textposition="top center"
    ))
    fig_risk.add_trace(go.Scatter(x=d_tot["Date"], y=d_tot["Peak"], name='Peak', line=dict(dash='dash', color='#9E9E9E')))
    fig_risk.add_trace(go.Scatter(x=d_tot["Date"], y=d_tot["Peak"]*0.93, name='-7%', line=dict(dash='dot', color='#FFB74D')))
    fig_risk.add_trace(go.Scatter(x=d_tot["Date"], y=d_tot["Peak"]*0.85, name='-15%', line=dict(dash='dot', color='#E53935')))
    st.plotly_chart(style_chart(fig_risk), use_container_width=True, config={'staticPlot': True})

# TAB 2: TRENDS
with t2:
    st.subheader("Account Performance (Sorted by Value)")
    
    for i, acct in enumerate(acct_order):
        with st.container():
            st.markdown(f"#### {acct}")
            acct_df = f_df[f_df["Account"] == acct]
            dy = acct_df.groupby("Date")[["Total Value","Cash","Margin Balance","YTD"]].sum().reset_index()
            
            # --- ACCOUNT KPI CARDS ---
            latest_acct = acct_df.iloc[-1]
            cur_val = latest_acct["Total Value"]
            cur_ytd = latest_acct["YTD"]
            
            k1, k2 = st.columns(2)
            with k1: kpi_card("Current Value", f"${cur_val:,.0f}")
            with k2: kpi_card("YTD Return", f"{cur_ytd:.1f}%")
            
            # --- CHART ---
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Bars (LIGHTER BLUE)
            fig.add_trace(go.Bar(
                x=dy["Date"], y=dy["Total Value"], name="Val", 
                marker_color="#90CAF9", 
                text=dy["Total Value"], texttemplate='%{y:.2s}', textposition='inside',
                textfont=dict(color="#0D47A1") 
            ), secondary_y=False)
            
            fig.add_trace(go.Bar(
                x=dy["Date"], y=dy["Cash"], name="Cash", 
                marker_color="#A5D6A7" 
            ), secondary_y=False)
            
            # Line
            fig.add_trace(go.Scatter(
                x=dy["Date"], y=dy["YTD"], name="YTD %", 
                mode="lines+markers+text",
                line=dict(color='#2E7D32', width=3),
                marker=dict(size=8),
                text=dy["YTD"],
                texttemplate='%{y:.1f}%', 
                textposition="top center",
                textfont=dict(color="#2E7D32", size=12)
            ), secondary_y=True)
            
            fig = style_chart(fig)
            fig.update_layout(barmode='group')
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
            st.divider()

# TAB 3: ALLOCATION
with t3:
    c_l, c_r = st.columns([1, 1])
    with c_l:
        st.subheader("Visual Breakdown")
        fig_s = px.sunburst(latest, path=['Bucket', 'Account'], values='Total Value')
        fig_s.update_traces(textinfo="label+percent entry")
        st.plotly_chart(style_chart(fig_s), use_container_width=True, config={'staticPlot': True})
    with c_r:
        st.subheader("Data Table")
        piv = latest.groupby("Bucket")[["Total Value","Cash"]].sum().sort_values("Total Value", ascending=False).reset_index()
        piv["%"] = (piv["Total Value"]/val)*100
        
        # PLOTLY TABLE
        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=["<b>Bucket</b>", "<b>Total Value</b>", "<b>Cash</b>", "<b>Alloc %</b>"],
                fill_color='#E3F2FD',
                align='left',
                font=dict(color='black', size=14)
            ),
            cells=dict(
                values=[
                    piv.Bucket, 
                    piv["Total Value"].apply(lambda x: f"${x:,.0f}"), 
                    piv["Cash"].apply(lambda x: f"${x:,.0f}"), 
                    piv["%"].apply(lambda x: f"{x:.1f}%")
                ],
                fill_color='white',
                align='left',
                font=dict(color='black', size=12),
                height=30
            ))
        ])
        fig_table.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=400)
        st.plotly_chart(fig_table, use_container_width=True, config={'staticPlot': True})

# TAB 4: RISK
with t4:
    st.subheader("Account Drawdown Monitor")
    
    cols = st.columns(2)
    for i, acct in enumerate(acct_order):
        with cols[i % 2]:
            st.markdown(f"**{acct}**")
            hist = df[df["Account"] == acct].groupby("Date")["Total Value"].sum().reset_index()
            hist["Peak"] = hist["Total Value"].cummax()
            view = hist[(hist["Date"] >= pd.to_datetime(dr[0])) & (hist["Date"] <= pd.to_datetime(dr[1]))]
            
            if not view.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=view["Date"], y=view["Total Value"], name='Val', line=dict(color='#1976D2')))
                fig.add_trace(go.Scatter(x=view["Date"], y=view["Peak"], name='Peak', line=dict(dash='dash', color='#9E9E9E')))
                fig.add_trace(go.Scatter(x=view["Date"], y=view["Peak"]*0.93, name='-7%', line=dict(dash='dot', color='#FFB74D')))
                fig.add_trace(go.Scatter(x=view["Date"], y=view["Peak"]*0.85, name='-15%', line=dict(dash='dot', color='#E53935')))
                
                fig = style_chart(fig)
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
