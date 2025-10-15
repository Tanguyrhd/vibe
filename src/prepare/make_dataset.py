"""
Module for MBTI data preparation and cleaning.
This script handles:
- loading raw data,
- cleaning text (tweets, URLs, mentions, etc.),
- removing MBTI mentions,
- filtering by word count,
- and creating binary MBTI labels.
"""

import re
from pathlib import Path
import pandas as pd

# ==========================
#  CLEANING FUNCTIONS
# ==========================

def clean_posts(text: str) -> str:
    """
    Cleans an MBTI user's text:
    - Removes mentions, URLs, and special characters
    - Normalizes apostrophes and quotes
    - Converts to lowercase
    - Joins tweets with periods

    Args:
        text (str): Raw text containing multiple posts separated by "|||"

    Returns:
        str: Cleaned and concatenated text
    """
    tweets = text.split("|||")
    cleaned = []

    for tweet in tweets:
        tweet = re.sub(r'@\w+', '', tweet)                # Remove mentions
        tweet = re.sub(r'http\S+', '', tweet)             # Remove URLs
        tweet = tweet.replace('’', "'").replace('“', '"').replace('”', '"')
        tweet = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', '', tweet)  # Keep basic punctuation
        tweet = re.sub(r'\s+', ' ', tweet).strip()        # Clean spaces
        tweet = tweet.lower()

        if tweet:
            cleaned.append(tweet)

    return ". ".join(cleaned)


def remove_mbti_words(series: pd.Series) -> pd.Series:
    """
    Removes MBTI words (e.g., 'INTJ', 'ENFP') from a text column.

    Args:
        series (pd.Series): Column containing text.

    Returns:
        pd.Series: Cleaned column.
    """
    mbti_types = {
        "intj", "intp", "entj", "entp",
        "infj", "infp", "enfj", "enfp",
        "istj", "isfj", "estj", "esfj",
        "istp", "isfp", "estp", "esfp",
        "intjs", "intps", "entjs", "entps",
        "infjs", "infps", "enfjs", "enfps",
        "istjs", "isfjs", "estjs", "esfjs",
        "istps", "isfps", "estps", "esfps"
    }

    pattern = r'\b(?:' + '|'.join(re.escape(t) for t in mbti_types) + r')\b'
    series = series.str.replace(pattern, '', flags=re.IGNORECASE, regex=True)
    series = series.str.replace(r'\s+', ' ', regex=True).str.strip()
    return series


def filter_by_word_count(df: pd.DataFrame, col: str = "clean_text",
                         min_words: int = 1000, max_words: int = 1700) -> pd.DataFrame:
    """
    Filters users based on the length of their cleaned text.

    Args:
        df (pd.DataFrame): Data
        col (str): Name of the column to analyze
        min_words (int): Minimum number of words
        max_words (int): Maximum number of words

    Returns:
        pd.DataFrame: Filtered data
    """
    df = df[df[col].apply(lambda x: isinstance(x, str) and min_words < len(x.split()) < max_words)]
    return df.reset_index(drop=True)


def add_mbti_binary_columns(df: pd.DataFrame, col_name: str = "type") -> pd.DataFrame:
    """
    Creates 4 MBTI binary columns: EI, SN, TF, JP.

    Args:
        df (pd.DataFrame): Data containing the MBTI column
        col_name (str): Name of the MBTI column (e.g., "type")

    Returns:
        pd.DataFrame: Enriched data
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in dataframe.")

    df[col_name] = df[col_name].str.upper()
    valid_mask = df[col_name].str.match(r"^[E|I][S|N][T|F][J|P]$")
    df = df[valid_mask].reset_index(drop=True)

    df["EI"] = df[col_name].apply(lambda x: 1 if x[0] == 'E' else 0)
    df["SN"] = df[col_name].apply(lambda x: 1 if x[1] == 'S' else 0)
    df["TF"] = df[col_name].apply(lambda x: 1 if x[2] == 'T' else 0)
    df["JP"] = df[col_name].apply(lambda x: 1 if x[3] == 'J' else 0)

    return df


# ==========================
#  MAIN PIPELINE
# ==========================

def make_dataset(raw_path: str = "data/raw/mbti_1.csv",
                 processed_path: str = "data/processed/mbti_clean.csv") -> pd.DataFrame:
    """
    Full data preparation pipeline:
    1. Load data
    2. Clean text
    3. Remove MBTI words
    4. Filter by word count
    5. Create binary labels
    6. Save cleaned CSV

    Args:
        raw_path (str): Path to raw CSV
        processed_path (str): Path to save cleaned CSV

    Returns:
        pd.DataFrame: Cleaned dataset ready to use
    """
    print("Loading data...")
    df = pd.read_csv(raw_path)

    required_columns = {"posts", "type"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"❌ Missing required column(s) in CSV: {missing}. "
            f"Please make sure your raw data CSV contains the columns 'posts' and 'type'."
        )

    print("Cleaning posts...")
    df["clean_text"] = df["posts"].apply(clean_posts)

    print("Removing MBTI terms...")
    df["clean_text"] = remove_mbti_words(df["clean_text"])

    print("Filtering by text length...")
    df = filter_by_word_count(df)

    print("Adding binary columns...")
    df = add_mbti_binary_columns(df)

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Cleaned data saved → {processed_path}")

    return df


# ==========================
# INTERACTIVE RAW FILE SELECTION
# ==========================

if __name__ == "__main__":
    raw_folder = Path("data/raw")
    files = [f for f in raw_folder.glob("*.csv") if f.is_file()]

    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_folder}")

    print("Available files in data/raw/:")
    for i, f in enumerate(files):
        print(f"{i + 1}: {f.name}")

    # Ask user to choose
    while True:
        choice = input(f"Select the file to use (1-{len(files)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            raw_file = files[int(choice) - 1]
            break
        else:
            print("❌ Invalid input, try again.")

    processed_file = Path("data/processed") / f"{raw_file.stem}_clean.csv"
    make_dataset(raw_path=str(raw_file), processed_path=str(processed_file))
