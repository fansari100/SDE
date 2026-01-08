import pandas as pd


def schoolCount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process school-subject data to count schools offering specific subjects per state.

    Args:
        df: DataFrame with columns [school_id, state_code, subjects]
            - subjects: space-separated lowercase subject strings

    Returns:
        DataFrame with state_code and counts for english, maths, physics, chemistry
        in order of first state_code occurrence (after filtering)
    """
    # Work on a copy to avoid mutating input
    df = df.copy()

    # Split subjects into list and compute count
    df['_subjects_list'] = df['subjects'].str.split()
    df['_subject_count'] = df['_subjects_list'].str.len()

    # Filter: keep only schools offering at least 3 subjects
    df = df[df['_subject_count'] >= 3]

    # Clean state_code: retain only alphanumeric characters
    df['state_code'] = df['state_code'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

    # Vectorized membership check for target subjects
    subjects_to_check = ['english', 'maths', 'physics', 'chemistry']
    for subj in subjects_to_check:
        df[subj] = df['_subjects_list'].apply(lambda lst: subj in lst).astype(int)

    # Aggregate by state_code preserving first-encounter order
    result = (
        df.groupby('state_code', sort=False)[subjects_to_check]
        .sum()
        .reset_index()
    )

    return result
