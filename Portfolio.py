import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. SETUP ---
st.set_page_config(page_title="Portfolio Command Center", layout="wide")

# --- 🔒 SECURITY: LOGIN SYSTEM ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Please enter your password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again.
        st.text_input(
            "Password incorrect. Please enter your password", type="password", on_change=password_entered, key="password"
        )
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not run the rest of the app if password is wrong

# --- APP STARTS HERE (ONLY RUNS IF PASSWORD IS CORRECT) ---
st.title("🚀 Portfolio Command Center")

# --- 2. DATA LOADING (VIA SECRETS) ---
@st.cache_data(ttl=60)
def load_data():
    try:
        # Look for the link in Streamlit Secrets
        url = st.secrets["public_sheet_url"]
        if "PASTE_YOUR" in url: return pd.DataFrame()
        return pd.read_csv(url)
    except: return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("⚠️ Could not load data. Check your Secrets.")
    st.stop()

# --- 3. CLEANING ---
try:
    df.columns = df.columns.str.strip()
    df = df[[c for c in df.columns if "Unnamed" not in c]]
    df = df.dropna(subset=["Bucket"])

    # Clean Money Columns
    for col in ["Total Value", "Cash", "Margin Balance"]:
        if col in df.columns:
            s = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
            df[col] = pd.to_numeric(s.str.replace('-', '0'), errors='coerce').fillna(0)

    # Clean YTD Column
    if "YTD" in df.columns:
        s = df["YTD"].astype(str).str.replace('%', '', regex=False)
        df["YTD"] = pd.to_numeric(s, errors='coerce').fillna(0)
    else: df["YTD"] = 0.0

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.sort_values("Date")
    else:
        st.error("❌ 'Date' column missing."); st.stop()

except Exception as e:
    st.error(f"❌ Cleaning Error: {e}"); st.stop()

# --- 4. FILTERS ---
min_d, max_d = df["Date"].min(), df["Date"].max()
with st.sidebar:
    dr = st.date_input("Date Range", [min_d, max_d])

if len(dr) == 2:
    mask = (df["Date"] >= pd.to_datetime(dr[0])) & (df["Date"] <= pd.to_datetime(dr[1]))
    f_df = df.loc[mask]
else: f_df = df

if f_df.empty: st.warning("⚠️ No data in range."); st.stop()

# Metrics
latest = f_df[f_df["Date"] == f_df["Date"].max()]
# Sort Accounts by Value
acct_order = latest.groupby("Account")["Total Value"].sum().sort_values(ascending=False).index.tolist()
val = latest["Total Value"].sum()
cash = latest["Cash"].sum()
mrg = latest["Margin Balance"].sum()

# --- 5. TABS ---
t1, t2, t3, t4 = st.tabs(["📊 Summary", "📈 Trends (YTD)", "🥧 Alloc (Sunburst)", "⚠️ Risk"])

# TAB 1: SUMMARY
with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Net Liq", f"${val:,.0f}")
    c2.metric("Cash", f"${cash:,.0f}")
    c3.metric("Margin", f"${mrg:,.0f}", delta_color="inverse")
    st.divider()
    
    cl, cr = st.columns([2, 1])
    with cl:
        tr = f_df.groupby("Date")["Total Value"].sum().reset_index()
        fig_area = px.area(tr, x="Date", y="Total Value", template="plotly_white")
        st.plotly_chart(fig_area, use_container_width=True, key="h_tr")
    with cr:
        fig_sun = px.sunburst(latest, path=['Bucket', 'Account'], values='Total Value', template="plotly_white")
        st.plotly_chart(fig_sun, use_container_width=True, key="h_sun")

    # Global Risk
    d_tot = f_df.groupby("Date")["Total Value"].sum().reset_index()
    g_hist = df.groupby("Date")["Total Value"].sum().reset_index()
    g_hist["Peak"] = g_hist["Total Value"].cummax()
    d_tot = pd.merge(d_tot, g_hist[["Date", "Peak"]], on="Date", how="left")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d_tot["Date"], y=d_tot["Total Value"], name='Val', line=dict(color='#2962FF')))
    fig.add_trace(go.Scatter(x=d_tot["Date"], y=d_tot["Peak"], name='Peak', line=dict(dash='dash', color='green')))
    fig.add_trace(go.Scatter(x=d_tot["Date"], y=d_tot["Peak"]*0.93, name='-7%', line=dict(dash='dot', color='orange')))
    fig.add_trace(go.Scatter(x=d_tot["Date"], y=d_tot["Peak"]*0.85, name='-15%', line=dict(dash='dot', color='red')))
    fig.update_layout(template="plotly_white", title="Global Risk Monitor", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="h_ri")

# TAB 2: TRENDS (Sorted + YTD)
with t2:
    st.subheader("Account Performance (Sorted)")
    st.info("Line chart (Right Axis) now shows YTD % Return")
    
    for i, acct in enumerate(acct_order):
        acct_df = f_df[f_df["Account"] == acct]
        dy = acct_df.groupby("Date")[["Total Value","Cash","Margin Balance","YTD"]].sum().reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=dy["Date"], y=dy["Total Value"], name="Val", marker_color="#4285F4"), secondary_y=False)
        fig.add_trace(go.Bar(x=dy["Date"], y=dy["Cash"], name="Cash", marker_color="#34A853"), secondary_y=False)
        fig.add_trace(go.Bar(x=dy["Date"], y=dy["Margin Balance"], name="Mrg", marker_color="#EA4335"), secondary_y=False)
        fig.add_trace(go.Scatter(x=dy["Date"], y=dy["YTD"], name="YTD %", line=dict(color='#0F9D58', width=2)), secondary_y=True)
        
        fig.update_layout(height=400, title=acct, barmode='group', hovermode="x unified", template="plotly_white")
        fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
        fig.update_yaxes(title_text="YTD Return (%)", secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True, key=f"tr_{i}")
        st.divider()

# TAB 3: ALLOCATION
with t3:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Visual Breakdown (Sunburst)")
        fig_sun_l = px.sunburst(latest, path=['Bucket', 'Account'], values='Total Value', template="plotly_white")
        st.plotly_chart(fig_sun_l, use_container_width=True, key="al_sun")
    with c2:
        st.markdown("##### Data Breakdown")
        piv = latest.groupby("Bucket")[["Total Value","Cash"]].sum().sort_values("Total Value", ascending=False).reset_index()
        piv["%"] = (piv["Total Value"]/val)*100
        st.dataframe(piv, hide_index=True, use_container_width=True, 
            column_config={
                "Total Value": st.column_config.NumberColumn(format="$%.0f"),
                "Cash": st.column_config.NumberColumn(format="$%.0f"),
                "%": st.column_config.NumberColumn(format="%.1f%%")
            }
        )

# TAB 4: RISK (Sorted)
with t4:
    st.subheader("Account Risk Monitor (Sorted)")
    for i, acct in enumerate(acct_order):
        hist = df[df["Account"] == acct].groupby("Date")["Total Value"].sum().reset_index()
        hist["Peak"] = hist["Total Value"].cummax()
        
        view = hist[(hist["Date"] >= pd.to_datetime(dr[0])) & (hist["Date"] <= pd.to_datetime(dr[1]))]
        
        if not view.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=view["Date"], y=view["Total Value"], name='Val', line=dict(color='#4285F4')))
            fig.add_trace(go.Scatter(x=view["Date"], y=view["Peak"], name='Peak', line=dict(dash='dash', color='green')))
            fig.add_trace(go.Scatter(x=view["Date"], y=view["Peak"]*0.93, name='-7%', line=dict(dash='dot', color='orange')))
            fig.add_trace(go.Scatter(x=view["Date"], y=view["Peak"]*0.85, name='-15%', line=dict(dash='dot', color='red')))
            
            fig.update_layout(height=300, title=f"{acct}", hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True, key=f"ri_{i}")
            st.divider()