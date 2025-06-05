group_map = {
    'intj': 'NT', 'intp': 'NT', 'entj': 'NT', 'entp': 'NT',
    'infj': 'NF', 'infp': 'NF', 'enfj': 'NF', 'enfp': 'NF',
    'istj': 'SJ', 'isfj': 'SJ', 'estj': 'SJ', 'esfj': 'SJ',
    'istp': 'SP', 'isfp': 'SP', 'estp': 'SP', 'esfp': 'SP'
}

df['group'] = df['label'].map(lambda x: group_map.get(x.lower(), 'UNKNOWN'))
df = df[df['group'] != 'UNKNOWN']  # remove anything that slipped through