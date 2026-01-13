import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def run_policy_engine(enroll_df, demo_df, bio_df):
    # --- DATE NORMALIZATION ---
    for df in [enroll_df, demo_df, bio_df]:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year_month'] = df['date'].dt.to_period('M')

    # --- AGGREGATION ---
    enrol_agg = enroll_df.groupby(
        ['state', 'district', 'year_month'], as_index=False
    ).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    })

    enrol_agg['total_enrolments'] = (
        enrol_agg['age_0_5'] +
        enrol_agg['age_5_17'] +
        enrol_agg['age_18_greater']
    )

    demo_agg = demo_df.groupby(
        ['state', 'district', 'year_month'], as_index=False
    ).agg({
        'demo_age_5_17': 'sum',
        'demo_age_17_': 'sum'
    })

    demo_agg['total_demo_updates'] = (
        demo_agg['demo_age_5_17'] + demo_agg['demo_age_17_']
    )

    bio_agg = bio_df.groupby(
        ['state', 'district', 'year_month'], as_index=False
    ).agg({
        'bio_age_5_17': 'sum',
        'bio_age_17_': 'sum'
    })

    bio_agg['total_bio_updates'] = (
        bio_agg['bio_age_5_17'] + bio_agg['bio_age_17_']
    )

    # --- MASTER ---
    master = (
        enrol_agg
        .merge(demo_agg, on=['state', 'district', 'year_month'], how='left')
        .merge(bio_agg, on=['state', 'district', 'year_month'], how='left')
        .fillna(0)
    )

    # --- ASSI ---
    scaler = MinMaxScaler()
    master[['enrol_n', 'demo_n', 'bio_n']] = scaler.fit_transform(
        master[['total_enrolments', 'total_demo_updates', 'total_bio_updates']]
    )

    master['ASSI'] = (
        0.4 * master['enrol_n'] +
        0.3 * master['demo_n'] +
        0.3 * master['bio_n']
    )

    # --- STRESS METRICS ---
    master['update_burden_index'] = (
        (master['total_demo_updates'] + master['total_bio_updates']) /
        (master['total_enrolments'] + 1)
    ) * 1000

    master['stability_index'] = master.groupby(
        ['state', 'district']
    )['ASSI'].transform('std')

    master['bio_share'] = master['total_bio_updates'] / (
        master['total_bio_updates'] + master['total_demo_updates'] + 1
    )

    master['shift_index'] = master.groupby(
        ['state', 'district']
    )['bio_share'].diff().abs()

    # --- WARNINGS ---
    latest_month = master['year_month'].max()

    def explain(row):
        reasons = []
        if row['update_burden_index'] > 2000:
            reasons.append("High repeat updates")
        if row['shift_index'] > 0.4:
            reasons.append("Biometric–demographic shift")
        if row['stability_index'] > 0.2:
            reasons.append("Volatile demand")
        return " | ".join(reasons) if reasons else "General service stress"

    early_warnings = master[
        master['year_month'] == latest_month
    ].copy()

    early_warnings['auto_explanation'] = early_warnings.apply(explain, axis=1)

    # --- RANKING ---
    early_warnings['risk_score'] = (
        early_warnings['ASSI'] * 0.5 +
        early_warnings['shift_index'].fillna(0) * 0.3 +
        early_warnings['update_burden_index'] / 3000 * 0.2
    )

    early_warnings = early_warnings.sort_values(
        'risk_score', ascending=False
    )

    # --- CHILD BIOMETRIC HOTSPOTS ---
    child_bio = (
        bio_agg.groupby(['state', 'district'], as_index=False)
        .agg({'bio_age_5_17': 'sum'})
        .sort_values('bio_age_5_17', ascending=False)
    )

    return master, early_warnings, child_bio
