import pandas as pd
import re



def clean_posts(text):
    # Split the tweets on ||||
    tweets = text.split("|||")
    cleaned = []

    for tweet in tweets:
        # Remove mentions
        tweet = re.sub(r'@\w+', '', tweet)
        # Remove URLs
        tweet = re.sub(r'http\S+', '', tweet)
        # Normalize unicode quotes and dashes (optional)
        tweet = tweet.replace('’', "'").replace('“', '"').replace('”', '"')
        # Keep basic punctuation, remove other non-letter characters
        tweet = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', '', tweet)
        # Remove extra whitespace
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        # Lowercase
        tweet = tweet.lower()
        # Append only if not empty
        if tweet:
            cleaned.append(tweet)

    # Join tweets with a period and line break for readability
    return ". ".join(cleaned)



def stop_word(text):
    custom_stopwords = {
        "intj", "intp", "entj", "entp",
        "infj", "infp", "enfj", "enfp",
        "istj", "isfj", "estj", "esfj",
        "istp", "isfp", "estp", "esfp",
        "intjs", "intps", "entjs", "entps",
        "infjs", "infps", "enfjs", "enfps",
        "istjs", "isfjs", "estjs", "esfjs",
        "istps", "isfps", "estps", "esfps"
    }

    # Regex pattern to delete all MBTI related word
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in custom_stopwords) + r')\b'

    # Delete the words and extra space
    text = text.str.replace(pattern, '', flags=re.IGNORECASE, regex=True)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()

    return text




def filter_by_word_count(df, col='clean_text', min_words=1000, max_words=1700):
    df = df[df[col].apply(lambda x: min_words < len(x.split()) < max_words)].reset_index(drop=True)

    return df




def add_mbti_binary_columns(df, col_name='type'):

    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in dataframe.")

    if not df[col_name].apply(lambda x: isinstance(x, str) and len(x) == 4).all():
        raise ValueError(f"All values in '{col_name}' must be 4-character strings (MBTI types).")

    df[col_name] = df[col_name].str.upper()

    df['EI'] = df[col_name].apply(lambda x: 1 if x[0] == 'E' else 0)
    df['SN'] = df[col_name].apply(lambda x: 1 if x[1] == 'S' else 0)
    df['TF'] = df[col_name].apply(lambda x: 1 if x[2] == 'T' else 0)
    df['JP'] = df[col_name].apply(lambda x: 1 if x[3] == 'J' else 0)

    return df
