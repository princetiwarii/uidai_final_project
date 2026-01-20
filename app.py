import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import numpy as np
from engine import run_policy_engine



st.set_page_config(
    page_title="UIDAI Aadhaar Analytics Dashboard",
    layout="wide"
)

st.title("UIDAI Aadhaar Enrollment, Demographic & Biometric Analysis")
st.markdown("Hackathon Dashboard | Data-driven insights for policymakers")


def load_csv_folder(folder_path, category):
    all_files = os.listdir(folder_path)
    df_list = []

    for file in all_files:
        # Check if the file is a CSV AND contains the specific category keyword
        if file.endswith(".csv") and category.lower() in file.lower():
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            df_list.append(df)

    if not df_list:
        st.error(f"⚠️ No files found for category: {category}")
        return pd.DataFrame()  # Returns empty DF to prevent script crash

    return pd.concat(df_list, ignore_index=True)

# Load data
enroll_df = load_csv_folder("data", "enrolment")
demo_df = load_csv_folder("data", "demographic")
bio_df = load_csv_folder("data", "biometric")

@st.cache_data
def load_india_geojson():
    file_path = os.path.join(os.path.dirname(__file__), "india_states.geojson")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


india_geojson = load_india_geojson()



# --- INITIALIZE GLOBAL VARIABLES  ---
if 'enroll_forecast' not in st.session_state:
    st.session_state['enroll_forecast'] = 0

if 'demo_forecast' not in st.session_state:
    st.session_state['demo_forecast'] = 0

if 'bio_forecast' not in st.session_state:
    st.session_state['bio_forecast'] = 0

for df in [enroll_df, demo_df, bio_df]:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['state'] = df['state'].str.title()
    df['district'] = df['district'].str.title()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=250)
st.sidebar.header("🔍 Filters")

selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(enroll_df['state'].dropna().unique())
)

if selected_state != "All":
    enroll_df = enroll_df[enroll_df['state'] == selected_state]
    demo_df = demo_df[demo_df['state'] == selected_state]
    bio_df = bio_df[bio_df['state'] == selected_state]

selected_district = st.sidebar.selectbox(
    "Select District",
    ["All"] + sorted(enroll_df['district'].dropna().unique())
)

if selected_district != "All":
    enroll_df = enroll_df[enroll_df['district'] == selected_district]
    demo_df = demo_df[demo_df['district'] == selected_district]
    bio_df = bio_df[bio_df['district'] == selected_district]

min_date = enroll_df['date'].min()
max_date = enroll_df['date'].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

enroll_df = enroll_df[
    (enroll_df['date'] >= pd.to_datetime(date_range[0])) &
    (enroll_df['date'] <= pd.to_datetime(date_range[1]))
]
# Compute total fields
enroll_df['total_enrollment'] = (
    enroll_df['age_0_5'] +
    enroll_df['age_5_17'] +
    enroll_df['age_18_greater']
)

demo_df = demo_df[
    (demo_df['date'] >= pd.to_datetime(date_range[0])) &
    (demo_df['date'] <= pd.to_datetime(date_range[1]))
]

bio_df = bio_df[
    (bio_df['date'] >= pd.to_datetime(date_range[0])) &
    (bio_df['date'] <= pd.to_datetime(date_range[1]))
]

state_map_df = (
    enroll_df.groupby("state", as_index=False)
    .agg(total_enrollment=("total_enrollment", "sum"))
)


# ===============================
# POLICY & STRESS ANALYTICS ENGINE
# ===============================

master_df, early_warnings_df, child_bio_df = run_policy_engine(
    enroll_df.copy(),
    demo_df.copy(),
    bio_df.copy()
)

# 1. Data Health Status
st.sidebar.subheader("📊 System Health")
total_records = len(enroll_df) + len(demo_df) + len(bio_df)
st.sidebar.info(f"Records Processed: {total_records:,}")

# 2. Export Data Feature
st.sidebar.subheader("📥 Report Generation")
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(enroll_df)
st.sidebar.download_button(
    label="Download Filtered CSV",
    data=csv,
    file_name='uidai_analytics_report.csv',
    mime='text/csv',
)

st.sidebar.success("✅ Engine: v2.4 Stable")

st.sidebar.markdown("---")
st.sidebar.markdown("✅ UIDAI Dashboard v2.5")
st.sidebar.markdown(f"Updated: {pd.Timestamp.now().strftime('%d %b %Y')}")

# Compute total fields
enroll_df['total_enrollment'] = enroll_df['age_0_5'] + enroll_df['age_5_17'] + enroll_df['age_18_greater']
demo_df['total_demo_updates'] = demo_df['demo_age_5_17'] + demo_df['demo_age_17_']
bio_df['total_bio_updates'] = bio_df['bio_age_5_17'] + bio_df['bio_age_17_']

# Quick stats
st.write("### 📊 Quick Stats")
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Total Enrollments", f"{int(enroll_df['total_enrollment'].sum()):,}")
with kpi2:
    st.metric("Demo Updates", f"{int(demo_df['total_demo_updates'].sum()):,}")
with kpi3:
    st.metric("Bio Updates", f"{int(bio_df['total_bio_updates'].sum()):,}")

st.divider()

# =========================
# Tabs: 5 including Policy Alerts
# =========================
# tab1, tab2, tab3, tab4, tab5 ,tab6 = st.tabs([
#     "Enrollment Trends",
#     "Demographic Updates",
#     "Biometric Updates",
#     "Strategic Planning",
#     "🚨 Policy Alerts & Stress",
#     "🤖 UIDAI Smart Assistant"
# ])

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📝Enrollment Trends",
    "👥Demographic Updates",
    "👁️Biometric Updates",
    "👨‍✈️Strategic Planning",
    "🚨 Policy Alerts & Stress",
    "🤖 UIDAI Smart Assistant",
    "🗺️ India Map View"
])


# --- TAB 1: Enrollment Trends ---
with tab1:
    view = st.radio(
        "Select View",
        ["Monthly", "Daily"],
        horizontal=True,
        key="tab1_view"
    )

    enroll_df['total_enrollment'] = (
        enroll_df['age_0_5'] +
        enroll_df['age_5_17'] +
        enroll_df['age_18_greater']
    )

    if view == "Daily":
        st.subheader("Aadhaar Enrollment by Age Group")

        enroll_summary = enroll_df.groupby(
            'date', as_index=False
        )['total_enrollment'].sum()

        fig = px.line(
            enroll_summary,
            x='date',
            y='total_enrollment',
            title="Total Aadhaar Enrollment Over Time"
        )

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Insight:**  
        Enrollment trends help identify policy impact periods and outreach effectiveness.
        """)

        daily_change = enroll_summary['total_enrollment'].pct_change().mean() * 100

        if daily_change > 0:
            st.info(
                f"**Trend Analysis:** Daily Aadhaar enrollments show an average growth of "
                f"{daily_change:.2f}%, indicating steady operational throughput."
            )
        else:
            st.warning(
                f"**Trend Analysis:** Daily Aadhaar enrollments show an average decline of "
                f"{abs(daily_change):.2f}%, possibly due to temporary operational or access constraints."
            )
        st.subheader("Daily Enrollment Anomaly Detection")

        mean = enroll_summary['total_enrollment'].mean()
        std = enroll_summary['total_enrollment'].std()

        if std > 0:
            enroll_summary['z_score'] = (
                    (enroll_summary['total_enrollment'] - mean) / std
            )
        else:
            enroll_summary['z_score'] = 0

        enroll_summary['anomaly_flag'] = enroll_summary['z_score'].abs() > 2

        fig = px.line(
            enroll_summary,
            x='date',
            y='total_enrollment',
            title="Anomalous Daily Enrollment Patterns"
        )

        fig.add_scatter(
            x=enroll_summary[enroll_summary['anomaly_flag']]['date'],
            y=enroll_summary[enroll_summary['anomaly_flag']]['total_enrollment'],
            mode='markers',
            marker=dict(size=9,color='red'),
            name='Anomaly'
        )

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)
        if enroll_summary['anomaly_flag'].any():
            st.warning(
                "⚠️ **Anomaly Detected:** "
                "Certain days exhibit unusually high or low Aadhaar enrollment volumes "
                "compared to normal daily patterns. This may result from localized enrollment drives, "
                "system downtime, weather disruptions, or access limitations."
            )
        st.success(
            "✅ **Recommended Action:** "
            "Examine operational logs and enrollment center activity on flagged days. "
            "Ensure adequate staffing, system availability, and contingency planning "
            "to maintain consistent daily enrollment performance."
        )
        st.subheader("Flagged Anomalous Days")

        st.dataframe(
            enroll_summary[enroll_summary['anomaly_flag']]
            [['date', 'total_enrollment', 'z_score']]
            .sort_values(by='z_score', ascending=False)
            .reset_index(drop=True)
        )
        WINDOW = 7
        enroll_summary = enroll_summary.sort_values('date')

        # Rolling average for stable forecast
        enroll_summary['rolling_mean'] = (
            enroll_summary['total_enrollment']
            .rolling(WINDOW, min_periods=1)
            .mean()
        )

        forecast_days = 7
        last_date = enroll_summary['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({ 'date': future_dates, 'forecast_enrollment': [enroll_summary['rolling_mean'].iloc[-1]] * forecast_days })
        forecast = forecast_df['forecast_enrollment'].sum()
        fig = px.line(forecast_df, x='date', y='forecast_enrollment',
                      title="7-Day Enrollment Forecast")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📈 **7-Day Enrollment Forecast:** Projected Aadhaar enrollments indicate consistent operational throughput from Dec 12–17, 2025, "
            "aiding resource planning and load balancing.")


    else:
        st.subheader("Monthly Aadhaar Enrollment Trend")

        enroll_df['month'] = enroll_df['date'].dt.to_period('M')

        monthly_enroll = enroll_df.groupby(
            'month', as_index=False
        )['total_enrollment'].sum()

        monthly_enroll = monthly_enroll.sort_values('month')
        monthly_enroll['month'] = monthly_enroll['month'].astype(str)

        fig = px.line(
            monthly_enroll,
            x='month',
            y='total_enrollment',
            title="Monthly Aadhaar Enrollment Trend"
        )

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Insight:**  
        Monthly trends reveal the effectiveness of enrollment drives and seasonal policy impact.
        """)
        growth = monthly_enroll['total_enrollment'].pct_change().mean() * 100

        if growth > 0:
            st.success(
                f"**Trend Analysis:** Enrollment shows an average monthly growth of "
                f"{growth:.2f}%, indicating improving Aadhaar adoption."
            )
        else:
            st.error(
                f"**Trend Analysis:** Enrollment shows an average monthly decline of "
                f"{abs(growth):.2f}%, indicating need for policy intervention."
            )
        st.subheader("Enrollment Anomaly Detection")

        # Z-score based anomaly detection
        mean = monthly_enroll['total_enrollment'].mean()
        std = monthly_enroll['total_enrollment'].std()
        if std > 0:
            monthly_enroll['z_score'] = (
                (monthly_enroll['total_enrollment'] - mean) / std
            )
        else:
            monthly_enroll['z_score'] = 0
        monthly_enroll['anomaly_flag'] = monthly_enroll['z_score'].abs() > 2

        fig = px.line(
            monthly_enroll,
            x='month',
            y='total_enrollment',
            title="Anomalous Enrollment Patterns Over Time"
        )

        fig.add_scatter(
            x=monthly_enroll.loc[monthly_enroll['anomaly_flag'], 'month'],
            y=monthly_enroll.loc[monthly_enroll['anomaly_flag'], 'total_enrollment'],
            mode='markers',
            marker=dict(
                size=14,
                color='red',
                symbol='circle'
            ),
            name='Anomaly'
        )

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

        if monthly_enroll['anomaly_flag'].any():
            st.warning(
                "⚠️ **Anomaly Detected:** "
                "Certain periods show abnormal spikes or drops in Aadhaar enrollments "
                "compared to historical trends. This may reflect special enrollment drives, "
                "policy interventions, or temporary reporting irregularities."
            )

            st.success(
                "✅ **Recommended Action:** "
                "Review operational logs and policy events during flagged periods to "
                "distinguish genuine demand surges from data or process-related issues."
            )

        #  FORECAST
        st.subheader("📈 Enrollment Forecast")

        monthly_enroll_sorted = monthly_enroll.sort_values('month').reset_index(drop=True)

        if len(monthly_enroll_sorted) >= 2:
            x = np.arange(len(monthly_enroll_sorted))
            y = monthly_enroll_sorted['total_enrollment'].values

            # Linear regression
            coef = np.polyfit(x, y, 1)
            slope = coef[0]
            intercept = coef[1]

            # CRITICAL FIX: Forecast for next point (x[-1] + 1)
            next_x = x[-1] + 1  # This is the next time point
            next_month_forecast = slope * next_x + intercept

            # Handle negative forecasts intelligently
            if next_month_forecast < 0:
                # If trend predicts negative, use last 3 months average instead
                last_3_avg = monthly_enroll_sorted['total_enrollment'].tail(3).mean()
                next_month_forecast = int(last_3_avg)
                forecast_method = "3-month average (declining trend)"
            else:
                next_month_forecast = int(next_month_forecast)
                forecast_method = "linear trend"

            st.session_state['enroll_forecast'] = next_month_forecast

            # Calculate metrics
            last_month = monthly_enroll_sorted['total_enrollment'].iloc[-1]
            change_pct = ((next_month_forecast - last_month) / last_month * 100) if last_month > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Next Month Forecast",
                    f"{next_month_forecast:,}",
                    delta=f"{change_pct:+.1f}% vs last month"
                )
            with col2:
                st.metric("Last Month Actual", f"{int(last_month):,}")
            with col3:
                trend_icon = "📈" if slope > 0 else "📉"
                st.metric("Monthly Trend", f"{trend_icon} {abs(slope):,.0f}")

            st.caption(f"Method: {forecast_method} | Based on {len(monthly_enroll_sorted)} months of data")

        else:
            last_month = monthly_enroll_sorted['total_enrollment'].iloc[-1] if len(monthly_enroll_sorted) > 0 else 0
            st.session_state['enroll_forecast'] = int(last_month)
            st.metric("Next Month Forecast", f"{int(last_month):,}")
            st.warning("⚠️ Insufficient data for trend analysis - using last month's value")

        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write("Forecast stored:", st.session_state.get('enroll_forecast', 'Not set'))
            st.write("Data points used:", len(monthly_enroll_sorted))
            if len(monthly_enroll_sorted) >= 2:
                st.write(f"Slope: {slope:,.2f}")
                st.write(f"Intercept: {intercept:,.2f}")
                st.write(f"Next X value: {next_x}")
                st.write(f"Raw forecast: {slope * next_x + intercept:,.2f}")
                st.write("Last 5 months data:")
                st.dataframe(monthly_enroll_sorted[['month', 'total_enrollment']].tail())

        st.caption(
            "Forecasts are trend-based projections. Declining trends use 3-month average for stability."
        )

# --- TAB 2: Demographic Updates ---
with tab2:
    st.subheader("Demographic Update Heatmap")

    demo_df['month'] = demo_df['date'].dt.to_period('M').astype(str)

    demo_df['total_demo_updates'] = (
        demo_df['demo_age_5_17'] +
        demo_df['demo_age_17_']
    )

    heatmap_df = demo_df.groupby(
        ['state', 'month'],
        as_index=False
    )['total_demo_updates'].sum()

    fig = px.density_heatmap(
        heatmap_df,
        x='month',
        y='state',
        z='total_demo_updates',
        color_continuous_scale='Blues',
        title="State-wise Demographic Updates Heatmap"
    )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insight:**  
    Heatmap reveals seasonal demographic update patterns and highlights states with abnormal update frequency, supporting targeted interventions and policy planning.
    """)
    demo_summary = demo_df.groupby('state', as_index=False).agg(
        total_demo_updates=('total_demo_updates', 'sum')
    )
    top_demo_state = demo_summary.sort_values(
        by='total_demo_updates', ascending=False
    ).iloc[0]

    st.info(
        f"**Trend Analysis:** {top_demo_state['state']} leads in demographic updates, "
        "suggesting higher migration or address/mobile changes."
    )
    st.subheader("Demographic Update Anomaly Detection")
    anomaly_df = heatmap_df.copy()
    mean = anomaly_df['total_demo_updates'].mean()
    std = anomaly_df['total_demo_updates'].std()
    if std > 0:
         anomaly_df['z_score'] = (anomaly_df['total_demo_updates'] - mean) / std
    else:
        anomaly_df['z_score'] = 0
    anomaly_df['anomaly_flag'] = anomaly_df['z_score'].abs() > 2

    fig = px.scatter(
        anomaly_df, x='month', y='total_demo_updates',
        color='anomaly_flag',
        color_discrete_map={True: 'red', False: 'blue'},
        hover_data=['state', 'z_score']
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("Flagged Anomalous States & Periods")

    st.dataframe(
        anomaly_df[anomaly_df['anomaly_flag']]
        .sort_values(by='z_score', ascending=False)
        .reset_index(drop=True)
    )
    if anomaly_df['anomaly_flag'].any():
      st.warning("⚠️ **Anomaly Detected:** Unusual demographic update activity.This may indicate increased population mobility,"
      "large-scale address or contact detail changes, or the impact of targeted government initiatives.")

    #  RECOMMENDATION
    st.success(
        "✅ **Recommended Action:** "
        "Cross-validate demographic update spikes with migration statistics, welfare scheme rollouts, "
        "and administrative boundary changes. Deploy targeted outreach or temporary update facilities "
        "in high-activity areas to ensure service continuity and data accuracy."
    )
    # FORECAST
    st.subheader("📈 Demographic Update Forecast")

    demo_monthly = demo_df.groupby('month', as_index=False)['total_demo_updates'].sum()
    demo_monthly = demo_monthly.sort_values('month').reset_index(drop=True)

    if len(demo_monthly) >= 2:
        x = np.arange(len(demo_monthly))
        y = demo_monthly['total_demo_updates'].values

        coef = np.polyfit(x, y, 1)
        slope = coef[0]
        intercept = coef[1]

        # CRITICAL FIX: Use x[-1] + 1
        next_x = x[-1] + 1
        next_month_forecast = slope * next_x + intercept

        # Handle negative forecasts
        if next_month_forecast < 0:
            last_3_avg = demo_monthly['total_demo_updates'].tail(3).mean()
            next_month_forecast = int(last_3_avg)
            forecast_method = "3-month average (declining trend)"
        else:
            next_month_forecast = int(next_month_forecast)
            forecast_method = "linear trend"

        st.session_state['demo_forecast'] = next_month_forecast

        last_month = demo_monthly['total_demo_updates'].iloc[-1]
        change_pct = ((next_month_forecast - last_month) / last_month * 100) if last_month > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Next Month Forecast",
                f"{next_month_forecast:,}",
                delta=f"{change_pct:+.1f}% vs last month"
            )
        with col2:
            st.metric("Last Month Actual", f"{int(last_month):,}")
        with col3:
            trend_icon = "📈" if slope > 0 else "📉"
            st.metric("Monthly Trend", f"{trend_icon} {abs(slope):,.0f}")

        st.caption(f"Method: {forecast_method} | Based on {len(demo_monthly)} months of data")

    else:
        last_value = demo_monthly['total_demo_updates'].iloc[-1] if len(demo_monthly) > 0 else 0
        st.session_state['demo_forecast'] = int(last_value)
        st.metric("Next Month Forecast", f"{int(last_value):,}")
        st.warning("⚠️ Insufficient data - using last period's value")

    with st.expander("🔍 Debug Info"):
        st.write("Forecast stored:", st.session_state.get('demo_forecast', 'Not set'))
        st.write("Data points used:", len(demo_monthly))
        if len(demo_monthly) >= 2:
            st.write(f"Slope: {slope:,.2f}")
            st.write(f"Intercept: {intercept:,.2f}")

    st.caption("Forecasts are trend-based projections. Declining trends use 3-month average for stability.")

# --- TAB 3: Biometric Updates ---
with tab3:
    st.subheader("Biometric Update Trends")

    bio_df['total_bio_updates'] = (
        bio_df['bio_age_5_17'] +
        bio_df['bio_age_17_']
    )

    bio_summary = bio_df.groupby('district', as_index=False)[
        ['total_bio_updates']
    ].sum()

    fig = px.bar(
        bio_summary,
        x='district',
        y='total_bio_updates',
        title="District-wise Biometric Updates"
    )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**  
    Frequent biometric updates may indicate aging population or biometric quality challenges.
    """)

    top_bio_district = bio_summary.sort_values(
        by='total_bio_updates', ascending=False
    ).iloc[0]

    st.info(
        f"**Trend Analysis:** {top_bio_district['district']} shows high biometric updates, "
        "possibly due to aging population or biometric quality issues."
    )

    st.subheader("Biometric Update Anomaly Detection")

    bio_anomaly = bio_df.groupby(
        'district', as_index=False
    )['total_bio_updates'].sum()

    # Threshold = Top 5%
    threshold = bio_anomaly['total_bio_updates'].quantile(0.95)

    bio_anomaly['anomaly_flag'] = (
            bio_anomaly['total_bio_updates'] > threshold
    )

    # Plot
    fig = px.bar(
        bio_anomaly.sort_values('total_bio_updates', ascending=False).head(20),
        x='district',
        y='total_bio_updates',
        color='anomaly_flag',
        color_discrete_map={True: 'red', False: 'blue'},
        title="Districts with Unusually High Biometric Updates"
    )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
    if bio_anomaly['anomaly_flag'].any():
       st.warning("⚠️ **Anomaly Detected:** "
        "This district shows an unusually high number of biometric updates compared to normal trends. "
        "Possible reasons include biometric quality issues, repeated authentication failures, "
        "or aging-related biometric changes."
       )

    st.success(
        "✅ **Recommended Action:** "
        "Initiate biometric quality audits, recalibrate devices, and enhance operator training "
        "to reduce repeat biometric updates."
    )

    # FORECAST
    st.subheader("📈 Biometric Update Forecast")

    bio_df['month'] = bio_df['date'].dt.to_period('M').astype(str)
    bio_monthly = bio_df.groupby('month', as_index=False)['total_bio_updates'].sum()
    bio_monthly = bio_monthly.sort_values('month').reset_index(drop=True)

    if len(bio_monthly) >= 2:
        x = np.arange(len(bio_monthly))
        y = bio_monthly['total_bio_updates'].values

        coef = np.polyfit(x, y, 1)
        slope = coef[0]
        intercept = coef[1]

        # CRITICAL FIX: Use x[-1] + 1
        next_x = x[-1] + 1
        next_month_forecast = slope * next_x + intercept

        # Handle negative forecasts
        if next_month_forecast < 0:
            last_3_avg = bio_monthly['total_bio_updates'].tail(3).mean()
            next_month_forecast = int(last_3_avg)
            forecast_method = "3-month average (declining trend)"
        else:
            next_month_forecast = int(next_month_forecast)
            forecast_method = "linear trend"

        st.session_state['bio_forecast'] = next_month_forecast

        last_month = bio_monthly['total_bio_updates'].iloc[-1]
        change_pct = ((next_month_forecast - last_month) / last_month * 100) if last_month > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Next Month Forecast",
                f"{next_month_forecast:,}",
                delta=f"{change_pct:+.1f}% vs last month"
            )
        with col2:
            st.metric("Last Month Actual", f"{int(last_month):,}")
        with col3:
            trend_icon = "📈" if slope > 0 else "📉"
            st.metric("Monthly Trend", f"{trend_icon} {abs(slope):,.0f}")

        st.caption(f"Method: {forecast_method} | Based on {len(bio_monthly)} months of data")

    else:
        last_value = bio_monthly['total_bio_updates'].iloc[-1] if len(bio_monthly) > 0 else 0
        st.session_state['bio_forecast'] = int(last_value)
        st.metric("Next Month Forecast", f"{int(last_value):,}")
        st.warning("⚠️ Insufficient data - using last period's value")

    with st.expander("🔍 Debug Info"):
        st.write("Forecast stored:", st.session_state.get('bio_forecast', 'Not set'))
        st.write("Data points used:", len(bio_monthly))
        if len(bio_monthly) >= 2:
            st.write(f"Slope: {slope:,.2f}")
            st.write(f"Intercept: {intercept:,.2f}")
            st.write(f"Raw forecast: {slope * next_x + intercept:,.2f}")

    st.caption("Forecasts are trend-based projections. Declining trends use 3-month average for stability.")

     

# --- TAB 4: Strategic Planning ---
with tab4:
    st.header("📍 Prescriptive Resource & Logistics Planning")
    st.markdown("Calculate infrastructure requirements based on forecasted demand and operational capacity.")

    # Get forecasts from session state
    enroll_forecast = st.session_state.get('enroll_forecast', 0)
    demo_forecast = st.session_state.get('demo_forecast', 0)
    bio_forecast = st.session_state.get('bio_forecast', 0)

    # Check if any forecasts are available
    if enroll_forecast == 0 and demo_forecast == 0 and bio_forecast == 0:
        st.warning("⚠️ Please visit other tabs first to generate forecasts for resource planning.")
        st.info(
            "💡 Navigate to 'Enrollment Trends', 'Demographic Updates', and 'Biometric Updates' tabs to calculate projections.")
    else:
        # Determine geography
        geography_note = "India (All States)"
        geography_scale = "national"

        if selected_state != "All":
            geography_note = selected_state
            geography_scale = "state"
            if selected_district != "All":
                geography_note = f"{selected_district}, {selected_state}"
                geography_scale = "district"

        st.info(
            "💡 **Instructions:** Adjust the parameters below based on current field capacity to see required resources.")

        with st.expander("⚙️ Configure Operational Parameters", expanded=True):
            col_p1, col_p2, col_p3 = st.columns(3)

            updates_per_kit = col_p1.number_input(
                "Updates per Kit / Day",
                value=100,
                min_value=10,
                max_value=500,
                step=10,
                help="Average capacity of one Aadhaar Enrollment Kit per day. Typical: 80-150 updates/day"
            )

            working_days = col_p2.number_input(
                "Working Days per Month",
                value=22,
                min_value=15,
                max_value=30,
                help="Operational days excluding holidays"
            )

            personnel_per_kit = col_p3.number_input(
                "Personnel per Kit",
                value=2,
                min_value=1,
                max_value=5,
                help="Operators needed per enrollment station"
            )

            st.markdown("---")
            col_u1, col_u2 = st.columns(2)

            utilization_rate = col_u1.slider(
                "Target Utilization Rate (%)",
                min_value=50,
                max_value=95,
                value=80,
                help="80% allows buffer for peak demand and maintenance"
            ) / 100

            buffer_percentage = col_u2.slider(
                "Contingency Buffer (%)",
                min_value=0,
                max_value=20,
                value=10,
                help="Extra capacity for seasonal surges"
            )

        # Calculation Logic
        total_forecasted_load = enroll_forecast + demo_forecast + bio_forecast

        # Calculate requirements
        monthly_capacity_per_kit = updates_per_kit * working_days
        effective_capacity_per_kit = monthly_capacity_per_kit * utilization_rate

        if effective_capacity_per_kit > 0:
            kits_required = np.ceil(total_forecasted_load / effective_capacity_per_kit)
        else:
            kits_required = 0

        total_staff = kits_required * personnel_per_kit

        # Add buffer
        kits_with_buffer = int(kits_required * (1 + buffer_percentage / 100))
        staff_with_buffer = int(total_staff * (1 + buffer_percentage / 100))

        # Display forecast breakdown
        st.subheader(f"📊 Demand Forecast for {geography_note}")

        breakdown_col1, breakdown_col2, breakdown_col3, breakdown_col4 = st.columns(4)
        breakdown_col1.metric("📝 Enrollment", f"{enroll_forecast:,}")
        breakdown_col2.metric("👤 Demographic", f"{demo_forecast:,}")
        breakdown_col3.metric("👁️ Biometric", f"{bio_forecast:,}")
        breakdown_col4.metric("📊 **Total Load**", f"{int(total_forecasted_load):,}")

        st.divider()

        # Display resource calculations
        st.subheader(f"🎯 Required Infrastructure for {geography_note}")

        res_col1, res_col2, res_col3 = st.columns(3)

        res_col1.metric(
            "📦 Enrollment Kits (Base)",
            f"{int(kits_required):,}",
            help=f"Based on {updates_per_kit} updates/day × {working_days} days × {utilization_rate * 100:.0f}% utilization"
        )

        res_col2.metric(
            "📦 Kits (with {buffer_percentage}% buffer)",
            f"{kits_with_buffer:,}",
            delta=f"+{kits_with_buffer - int(kits_required)}",
            help="Recommended capacity including contingency buffer"
        )

        res_col3.metric(
            "👥 Field Personnel Required",
            f"{staff_with_buffer:,}",
            help=f"{personnel_per_kit} operators per kit including buffer"
        )

        # Capacity Analysis
        st.divider()
        st.subheader("📈 Operational Metrics")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        # Daily metrics
        daily_load = total_forecasted_load / working_days
        daily_kits_needed = kits_required / 1  # Assume distributed evenly

        metric_col1.metric(
            "Daily Load Average",
            f"{int(daily_load):,}",
            help="Average updates per working day"
        )

        metric_col2.metric(
            "Updates per Kit/Month",
            f"{int(total_forecasted_load / kits_required if kits_required > 0 else 0):,}",
            help="Workload distributed per kit"
        )

        metric_col3.metric(
            "Updates per Kit/Day",
            f"{int((total_forecasted_load / kits_required) / working_days if kits_required > 0 else 0)}",
            help="Daily workload per kit"
        )

        if kits_required > 0:
            actual_util = (total_forecasted_load / (kits_required * monthly_capacity_per_kit)) * 100
            metric_col4.metric(
                "Projected Utilization",
                f"{actual_util:.1f}%",
                help="Efficiency of resource allocation"
            )

        # Context-aware recommendations
        st.divider()

        # Scale-appropriate messaging
        if geography_scale == "national":
            # All India perspective
            avg_per_state = kits_with_buffer / 28  # Approximate 28 states

            st.success(
                f"✅ **National Infrastructure Plan:** To meet the projected demand across India, "
                f"UIDAI should ensure **{kits_with_buffer:,} functional enrollment stations** "
                f"are operational nationwide. This translates to approximately **{int(avg_per_state)} kits per state** "
                f"with **{staff_with_buffer:,} trained personnel** to maintain service quality."
            )

            st.info(
                f"📊 **Context:** With a monthly load of **{int(total_forecasted_load):,}** updates across India, "
                f"this represents approximately **{(total_forecasted_load / 1_400_000_000 * 100):.3f}%** of the population "
                f"requiring Aadhaar services each month."
            )

        elif geography_scale == "state":
            st.success(
                f"✅ **State Infrastructure Plan:** To meet the projected demand in {geography_note}, "
                f"ensure **{kits_with_buffer:,} functional enrollment stations** with "
                f"**{staff_with_buffer:,} personnel** are operational."
            )

        else:  # district
            st.success(
                f"✅ **District Infrastructure Plan:** To meet the projected demand in {geography_note}, "
                f"ensure **{kits_with_buffer:,} functional enrollment stations** with "
                f"**{staff_with_buffer:,} personnel** are operational."
            )

        # Buffer recommendation
        st.info(
            f"📋 **Contingency Planning:** The {buffer_percentage}% buffer ({kits_with_buffer - int(kits_required)} additional kits) "
            f"accounts for seasonal surges, equipment downtime, maintenance cycles, and peak demand periods during "
            f"enrollment drives or policy changes."
        )

        # Cost estimation
        with st.expander("💰 Budget Estimation"):
            st.markdown("### Infrastructure & Operational Costs")

            col_c1, col_c2, col_c3 = st.columns(3)

            cost_per_kit = col_c1.number_input(
                "Cost per Kit (₹)",
                value=50000,
                step=5000,
                help="Approximate hardware/software cost"
            )

            monthly_salary = col_c2.number_input(
                "Monthly Salary (₹/person)",
                value=25000,
                step=1000,
                help="Average compensation per operator"
            )

            annual_maintenance = col_c3.number_input(
                "Annual Maintenance (%)",
                value=15,
                min_value=5,
                max_value=30,
                help="% of equipment cost for maintenance"
            )

            # Calculations
            total_capex = kits_with_buffer * cost_per_kit
            monthly_opex = staff_with_buffer * monthly_salary
            annual_opex = monthly_opex * 12
            annual_maintenance_cost = total_capex * (annual_maintenance / 100)
            total_annual_cost = annual_opex + annual_maintenance_cost

            # Display costs
            st.markdown("---")
            cost_m1, cost_m2, cost_m3 = st.columns(3)

            cost_m1.metric(
                "💰 Capital Expenditure",
                f"₹{total_capex / 10000000:.2f} Cr",
                help=f"One-time equipment cost: ₹{total_capex:,.0f}"
            )

            cost_m2.metric(
                "💸 Annual Operating Cost",
                f"₹{annual_opex / 10000000:.2f} Cr",
                help=f"Personnel costs: ₹{annual_opex:,.0f}/year"
            )

            cost_m3.metric(
                "🔧 Annual Maintenance",
                f"₹{annual_maintenance_cost / 10000000:.2f} Cr",
                help=f"Equipment maintenance: ₹{annual_maintenance_cost:,.0f}/year"
            )

            # Efficiency metrics
            st.markdown("---")
            st.markdown("### Cost Efficiency Metrics")

            eff_col1, eff_col2, eff_col3 = st.columns(3)

            cost_per_update = (monthly_opex + (annual_maintenance_cost / 12)) / total_forecasted_load
            eff_col1.metric(
                "Cost per Update",
                f"₹{cost_per_update:.2f}",
                help="Average cost to process one update"
            )

            revenue_per_kit = (total_forecasted_load / kits_with_buffer) * cost_per_update * 12
            eff_col2.metric(
                "Annual Throughput/Kit",
                f"₹{revenue_per_kit / 100000:.2f} L",
                help="Value processed per kit annually"
            )

            roi_years = total_capex / (
                        revenue_per_kit * kits_with_buffer - total_annual_cost) if revenue_per_kit * kits_with_buffer > total_annual_cost else 0
            eff_col3.metric(
                "Payback Period",
                f"{roi_years:.1f} years" if roi_years > 0 else "N/A",
                help="Time to recover capital investment"
            )

        # Deployment Strategy
        with st.expander("🗺️ Deployment Strategy"):
            st.markdown(f"### Recommended Rollout Plan for {geography_note}")

            if geography_scale == "national":
                st.markdown("""
                **Phase 1: High-Density States (Months 1-3)**
                - Deploy 40% of kits in top 5 high-population states
                - Focus: Urban centers with high transaction volumes

                **Phase 2: Medium-Density States (Months 4-6)**
                - Deploy 35% of kits in tier-2 states
                - Focus: District headquarters and major towns

                **Phase 3: Rural & Remote Areas (Months 7-9)**
                - Deploy remaining 25% in rural and remote regions
                - Focus: Mobile enrollment units and periodic camps

                **Phase 4: Buffer & Optimization (Months 10-12)**
                - Deploy buffer capacity based on demand patterns
                - Optimize based on first 9 months' data
                """)
            else:
                phase1_kits = int(kits_with_buffer * 0.6)
                phase2_kits = int(kits_with_buffer * 0.3)
                phase3_kits = kits_with_buffer - phase1_kits - phase2_kits

                st.markdown(f"""
                **Phase 1: Urban Centers (Month 1-2)** → {phase1_kits} kits
                - Deploy in major cities and district headquarters

                **Phase 2: Semi-Urban Areas (Month 3-4)** → {phase2_kits} kits
                - Deploy in towns and block offices

                **Phase 3: Rural Areas (Month 5-6)** → {phase3_kits} kits
                - Deploy in villages and remote areas
                - Consider mobile enrollment units
                """)

        # Data quality check
        if total_forecasted_load > 50_000_000:  # 50 million for All India
            st.warning(
                "⚠️ **Data Validation Notice:** Monthly load exceeds 50 million updates for All India. "
                "While possible during major enrollment drives, please verify forecast accuracy."
            )
        elif total_forecasted_load < 100_000 and geography_scale == "national":
            st.warning(
                "⚠️ **Low Volume Alert:** Monthly load seems unusually low for national scope. "
                "Verify date range and data completeness."
            )

# --- TAB 5: Policy Alerts & Stress ---
with tab5:
    st.header("🚨 Aadhaar Service Stress & Early Warning System")

    st.subheader("🔴 High-Risk Districts (Latest Month)")
    st.dataframe(
        early_warnings_df[
            ['state', 'district', 'ASSI', 'risk_score', 'auto_explanation']
        ].head(10)
    )

    st.subheader("🧒 Child Biometric Failure Hotspots")
    st.caption("High biometric updates for age group 5–17")
    st.dataframe(
        child_bio_df.head(10)
    )

    st.subheader("📊 Stress Index Distribution")
    fig = px.histogram(
        master_df,
        x='ASSI',
        nbins=30,
        title="Aadhaar Service Stress Index (ASSI)"
    )
    st.plotly_chart(fig, use_container_width=True)

import google.generativeai as genai

# ==========================================
# 🤖 REAL AI SIDEBAR CHATBOT (GEMINI)
# ==========================================
# --- TAB 6: UIDAI SMART ASSISTANT ---
with tab6:

    st.markdown("## 🤖 UIDAI Policy Intelligence Assistant")

    # Setup Gemini
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBxBuos0Gx8kSSYFKWAbZD0sKH5wuoRJDU"
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Suggested Questions
    st.markdown("### 💡 Suggested Policy Questions")
    suggested_questions = [
        "Why is this state showing high Aadhaar service stress?",
        "Which districts need immediate infrastructure support?",
        "What is driving biometric update spikes?",
        "What actions should UIDAI take this month?",
        "Is this trend temporary or structural?",
    ]

    cols = st.columns(len(suggested_questions))
    for i, q in enumerate(suggested_questions):
        if cols[i].button(q, key=f"suggest_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "I am a UIDAI Policy Intelligence Assistant. "
                "I analyze Aadhaar enrollment, demographic, and biometric trends "
                "to support policymakers."
            )
        }]

    # Chat display (INSIDE TAB)
    chat_box = st.container(height=500)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Floating input
    prompt = st.chat_input("Ask a policy or operational question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        full_context = f"""
You are a UIDAI Policy Intelligence Assistant.

State: {selected_state}
District: {selected_district}
Total Enrollments: {int(enroll_df['total_enrollment'].sum()):,}
High Stress Districts: {len(early_warnings_df)}

Question:
{prompt}
"""

        try:
            response = model.generate_content(full_context)
            ai_response = response.text
        except Exception as e:
            ai_response = f"⚠️ AI error: {e}"

        st.session_state.messages.append(
            {"role": "assistant", "content": ai_response}
        )

        st.rerun()




with tab7:
    st.subheader("🗺️ State-wise Aadhaar Activity Map")

    fig = px.choropleth(
        state_map_df,
        geojson=india_geojson,
        locations="state",
        featureidkey="properties.st_nm",
        color="total_enrollment",
        color_continuous_scale="Oranges",
        title="Total Enrollments by State"
    )

    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "📌 Darker states indicate higher Aadhaar enrollment activity."
    )