"""
Portfolio Command Center - Main Application
A comprehensive portfolio tracking dashboard with risk monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Command Center", layout="wide", page_icon="🚀")

# --- CONSTANTS ---
CHART_CONFIG = {'displayModeBar': False, 'staticPlot': False}
WARNING_THRESHOLD = 0.93  # -7%
DANGER_THRESHOLD = 0.85   # -15%
DATA_CACHE_TTL = 60  # seconds

# --- CSS STYLING ---
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

# --- SECURITY ---
def check_password():
    """
    Password authentication for the dashboard.
    Returns True if authenticated, False otherwise.
    """
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    password = st.text_input("🔐 Enter Password", type="password", key="password_input")
    
    if password:
        try:
            if password == st.secrets["app_password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        except KeyError:
            st.error("⚠️ Password not configured in secrets")
    
    return False

if not check_password():
    st.stop()

# --- HELPER FUNCTIONS ---
def kpi_card(label, value):
    """
    Display a styled KPI card with label and value.
    
    Args:
        label (str): The label for the KPI
        value (str): The formatted value to display
    """
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def style_chart(fig):
    """
    Apply consistent styling to Plotly charts.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        Styled Plotly figure
    """
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
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#E0E0E0", gridwidth=1,
            showline=False, zeroline=False,
            tickfont=dict(color="#1F2937"),
            fixedrange=True
        ),
        yaxis2=dict(
            showgrid=False, showline=False, zeroline=False,
            tickfont=dict(color="#1F2937"),
            overlaying="y", side="right",
            fixedrange=True
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#1F2937"), bgcolor="rgba(255,255,255,0.8)"
        ),
        hovermode="x unified"
    )
    return fig

def create_drawdown_chart(data, date_col="Date", value_col="Total Value", height=None, show_legend=True):
    """
    Create a drawdown monitoring chart with peak and threshold lines.
    
    Args:
        data (pd.DataFrame): DataFrame with date and value columns
        date_col (str): Name of the date column
        value_col (str): Name of the value column
        height (int, optional): Chart height in pixels
        show_legend (bool): Whether to show the legend
        
    Returns:
        Plotly figure object
    """
    data = data.sort_values(date_col).copy()
    data["Peak"] = data[value_col].cummax()
    
    fig = go.Figure()
    
    # Value line
    fig.add_trace(go.Scatter(
        x=data[date_col], 
        y=data[value_col], 
        name='Value', 
        line=dict(color='#1976D2', width=3),
        mode="lines+markers",
        marker=dict(size=6)
    ))
    
    # Peak line
    fig.add_trace(go.Scatter(
        x=data[date_col], 
        y=data["Peak"], 
        name='Peak', 
        line=dict(dash='dash', color='#9E9E9E', width=2)
    ))
    
    # Warning threshold (-7%)
    fig.add_trace(go.Scatter(
        x=data[date_col], 
        y=data["Peak"] * WARNING_THRESHOLD, 
        name='-7%', 
        line=dict(dash='dot', color='#FFB74D', width=2)
    ))
    
    # Danger threshold (-15%)
    fig.add_trace(go.Scatter(
        x=data[date_col], 
        y=data["Peak"] * DANGER_THRESHOLD, 
        name='-15%', 
        line=dict(dash='dot', color='#E53935', width=2)
    ))
    
    fig = style_chart(fig)
    
    if height:
        fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0))
    
    if not show_legend:
        fig.update_layout(showlegend=False)
    
    return fig

# --- DATA LOADING ---
@st.cache_data(ttl=DATA_CACHE_TTL)
def load_data():
    """
    Load portfolio data from Google Sheets.
    
    Returns:
        pd.DataFrame: Loaded data or empty DataFrame on error
    """
    try:
        url = st.secrets.get("public_sheet_url")
        if not url or "PASTE_YOUR" in url:
            st.error("⚠️ Google Sheet URL not configured in secrets")
            return pd.DataFrame()
        
        with st.spinner('📊 Loading portfolio data...'):
            df = pd.read_csv(url)
        return df
        
    except KeyError:
        st.error("⚠️ Missing 'public_sheet_url' in secrets. Please configure in Streamlit settings.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return pd.DataFrame()

def clean_data(df):
    """
    Clean and normalize the portfolio data.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    if df.empty:
        return df
    
    # Clean column names
    df.columns = df.columns.str.strip()
    df = df[[c for c in df.columns if "Unnamed" not in c]]
    
    # Remove rows without bucket assignment
    if "Bucket" in df.columns:
        df = df.dropna(subset=["Bucket"])
    else:
        st.error("⚠️ Missing required 'Bucket' column")
        return pd.DataFrame()
    
    # Clean currency columns
    currency_cols = ["Total Value", "Cash", "Margin Balance"]
    for col in currency_cols:
        if col in df.columns:
            s = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
            df[col] = pd.to_numeric(s.str.replace('-', '0'), errors='coerce').fillna(0)
    
    # Clean percentage column
    if "YTD" in df.columns:
        s = df["YTD"].astype(str).str.replace('%', '', regex=False)
        df["YTD"] = pd.to_numeric(s, errors='coerce').fillna(0)
    else:
        df["YTD"] = 0.0
    
    # Clean date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date"])  # Remove rows with invalid dates
        df = df.sort_values("Date")
    else:
        st.error("⚠️ Missing required 'Date' column")
        return pd.DataFrame()
    
    return df

def validate_data(df):
    """
    Validate the data has required columns and structure.
    
    Args:
        df (pd.DataFrame): Cleaned data
        
    Returns:
        bool: True if valid, False otherwise
    """
    if df.empty:
        st.error("❌ No data loaded. Please check your data source.")
        return False
    
    required_cols = ["Date", "Bucket", "Account", "Total Value"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return False
    
    return True

# --- LOAD AND VALIDATE DATA ---
df = load_data()
df = clean_data(df)

if not validate_data(df):
    st.stop()

# Display data quality info in sidebar
with st.sidebar:
    with st.expander("📋 Data Quality"):
        st.metric("Total Rows", len(df))
        st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    
    st.caption(f"🔄 Data refreshes every {DATA_CACHE_TTL} seconds")

# --- FILTERS ---
with st.sidebar:
    st.header("🔍 Filters")
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    
    # Set default start date to Jan 1, 2026 or earliest available
    default_start = date(2026, 1, 1)
    if default_start < min_date:
        default_start = min_date
    
    date_range = st.date_input(
        "Date Range", 
        [default_start, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Apply date filter
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df["Date"] >= pd.to_datetime(start_date)) & 
        (df["Date"] <= pd.to_datetime(end_date))
    ]
elif len(date_range) == 1:
    st.warning("⚠️ Please select both start and end dates")
    filtered_df = df
else:
    filtered_df = df

# Check if filter produced results
if filtered_df.empty:
    st.warning("📭 No data found for selected date range. Try expanding your filter.")
    st.stop()

# Calculate key metrics
latest = filtered_df[filtered_df["Date"] == filtered_df["Date"].max()]
account_order = latest.groupby("Account")["Total Value"].sum().sort_values(ascending=False).index.tolist()
total_value = latest["Total Value"].sum()
total_cash = latest["Cash"].sum()
total_margin = latest["Margin Balance"].sum()

# --- DASHBOARD ---
st.title("🚀 Portfolio Command Center")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_card("Net Liquidity", f"${total_value:,.0f}")
with col2:
    kpi_card("Cash", f"${total_cash:,.0f}")
with col3:
    kpi_card("Margin", f"${total_margin:,.0f}")
with col4:
    kpi_card("Active Accounts", f"{len(account_order)}")

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "📈 Trends (YTD)", 
    "🥧 Allocation", 
    "⚠️ Risk Monitor"
])

# TAB 1: EXECUTIVE SUMMARY
with tab1:
    col_trend, col_sun = st.columns([2, 1])
    
    with col_trend:
        st.subheader("Portfolio Growth")
        try:
            trend_data = filtered_df.groupby("Date")["Total Value"].sum().reset_index()
            
            fig = px.area(trend_data, x="Date", y="Total Value")
            fig.update_traces(
                mode="lines+markers+text", 
                line_color="#1565C0", 
                fillcolor="rgba(21, 101, 192, 0.1)",
                marker=dict(size=6),
                text=trend_data["Total Value"],
                texttemplate='%{y:.2s}',
                textposition="top center",
                textfont=dict(size=12, color="#1565C0")
            )
            
            st.plotly_chart(
                style_chart(fig), 
                use_container_width=True, 
                config=CHART_CONFIG, 
                key="summ_trend"
            )
        except Exception as e:
            st.error(f"Error creating trend chart: {str(e)}")

    with col_sun:
        st.subheader("Allocation Hierarchy")
        try:
            fig_sun = px.sunburst(
                latest, 
                path=['Bucket', 'Account'], 
                values='Total Value'
            )
            fig_sun.update_traces(textinfo="label+percent entry")
            
            st.plotly_chart(
                style_chart(fig_sun), 
                use_container_width=True, 
                config=CHART_CONFIG, 
                key="summ_sun"
            )
        except Exception as e:
            st.error(f"Error creating sunburst chart: {str(e)}")

    # Global Risk Monitor
    st.subheader("Global Risk Monitor")
    try:
        daily_totals = filtered_df.groupby("Date")["Total Value"].sum().reset_index()
        
        fig_risk = create_drawdown_chart(
            daily_totals, 
            date_col="Date", 
            value_col="Total Value",
            show_legend=True
        )
        
        # Add value labels
        fig_risk.update_traces(
            text=daily_totals["Total Value"],
            texttemplate='%{y:.2s}',
            textposition="top center",
            textfont=dict(color="#1565C0"),
            selector=dict(name="Value")
        )
        
        st.plotly_chart(
            fig_risk, 
            use_container_width=True, 
            config=CHART_CONFIG, 
            key="summ_risk"
        )
    except Exception as e:
        st.error(f"Error creating risk chart: {str(e)}")

# TAB 2: TRENDS
with tab2:
    st.subheader("Account Performance (Sorted by Value)")
    
    for i, account in enumerate(account_order):
        try:
            with st.container():
                st.markdown(f"#### {account}")
                
                account_df = filtered_df[filtered_df["Account"] == account]
                daily_account = account_df.groupby("Date")[
                    ["Total Value", "Cash", "Margin Balance", "YTD"]
                ].sum().reset_index()
                
                # Get latest values for this account
                latest_account = account_df.iloc[-1]
                current_value = latest_account["Total Value"]
                current_ytd = latest_account["YTD"]
                
                # KPI Cards
                kpi1, kpi2 = st.columns(2)
                with kpi1:
                    kpi_card("Current Value", f"${current_value:,.0f}")
                with kpi2:
                    kpi_card("YTD Return", f"{current_ytd:.1f}%")
                
                # Create dual-axis chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Bar: Total Value
                fig.add_trace(
                    go.Bar(
                        x=daily_account["Date"], 
                        y=daily_account["Total Value"], 
                        name="Value", 
                        marker_color="#90CAF9", 
                        text=daily_account["Total Value"], 
                        texttemplate='%{y:.2s}', 
                        textposition='inside',
                        textfont=dict(color="#0D47A1")
                    ), 
                    secondary_y=False
                )
                
                # Bar: Cash
                fig.add_trace(
                    go.Bar(
                        x=daily_account["Date"], 
                        y=daily_account["Cash"], 
                        name="Cash", 
                        marker_color="#A5D6A7"
                    ), 
                    secondary_y=False
                )
                
                # Line: YTD %
                fig.add_trace(
                    go.Scatter(
                        x=daily_account["Date"], 
                        y=daily_account["YTD"], 
                        name="YTD %", 
                        mode="lines+markers+text",
                        line=dict(color='#2E7D32', width=3),
                        marker=dict(size=8),
                        text=daily_account["YTD"],
                        texttemplate='%{y:.1f}%', 
                        textposition="top center",
                        textfont=dict(color="#2E7D32", size=12)
                    ), 
                    secondary_y=True
                )
                
                fig = style_chart(fig)
                fig.update_layout(barmode='group')
                fig.update_yaxes(title_text="Value ($)", secondary_y=False)
                fig.update_yaxes(title_text="YTD (%)", secondary_y=True)
                
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    config=CHART_CONFIG, 
                    key=f"trend_{i}"
                )
                st.divider()
                
        except Exception as e:
            st.error(f"Error creating chart for {account}: {str(e)}")
            continue

# TAB 3: ALLOCATION
with tab3:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Visual Breakdown")
        try:
            fig_sunburst = px.sunburst(
                latest, 
                path=['Bucket', 'Account'], 
                values='Total Value'
            )
            fig_sunburst.update_traces(textinfo="label+percent entry")
            
            st.plotly_chart(
                style_chart(fig_sunburst), 
                use_container_width=True, 
                config=CHART_CONFIG, 
                key="alloc_sun"
            )
        except Exception as e:
            st.error(f"Error creating allocation chart: {str(e)}")
    
    with col_right:
        st.subheader("Data Table")
        try:
            pivot = latest.groupby("Bucket")[["Total Value", "Cash"]].sum()
            pivot = pivot.sort_values("Total Value", ascending=False).reset_index()
            pivot["Allocation %"] = (pivot["Total Value"] / total_value) * 100
            
            fig_table = go.Figure(data=[go.Table(
                header=dict(
                    values=["<b>Bucket</b>", "<b>Total Value</b>", "<b>Cash</b>", "<b>Alloc %</b>"],
                    fill_color='#E3F2FD',
                    align='left',
                    font=dict(color='black', size=14)
                ),
                cells=dict(
                    values=[
                        pivot["Bucket"], 
                        pivot["Total Value"].apply(lambda x: f"${x:,.0f}"), 
                        pivot["Cash"].apply(lambda x: f"${x:,.0f}"), 
                        pivot["Allocation %"].apply(lambda x: f"{x:.1f}%")
                    ],
                    fill_color='white',
                    align='left',
                    font=dict(color='black', size=12),
                    height=30
                ))
            ])
            
            fig_table.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), 
                height=400
            )
            
            st.plotly_chart(
                fig_table, 
                use_container_width=True, 
                config=CHART_CONFIG, 
                key="alloc_table"
            )
        except Exception as e:
            st.error(f"Error creating allocation table: {str(e)}")

# TAB 4: RISK MONITOR
with tab4:
    st.subheader("Account Drawdown Monitor")
    
    cols = st.columns(2)
    for i, account in enumerate(account_order):
        try:
            with cols[i % 2]:
                st.markdown(f"**{account}**")
                
                # Get historical data for this account
                account_history = df[df["Account"] == account].groupby("Date")["Total Value"].sum().reset_index()
                
                # Filter to selected date range
                if len(date_range) == 2:
                    account_view = account_history[
                        (account_history["Date"] >= pd.to_datetime(date_range[0])) & 
                        (account_history["Date"] <= pd.to_datetime(date_range[1]))
                    ]
                else:
                    account_view = account_history
                
                if not account_view.empty:
                    fig = create_drawdown_chart(
                        account_view,
                        date_col="Date",
                        value_col="Total Value",
                        height=250,
                        show_legend=False
                    )
                    
                    st.plotly_chart(
                        fig, 
                        use_container_width=True, 
                        config=CHART_CONFIG, 
                        key=f"risk_{i}"
                    )
                else:
                    st.info("No data available for selected date range")
                    
        except Exception as e:
            st.error(f"Error creating risk chart for {account}: {str(e)}")
            continue
