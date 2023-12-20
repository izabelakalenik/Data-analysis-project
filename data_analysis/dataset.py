import pandas as pd
from IPython.core.display_functions import display

"""
The impact of listening music genres on mental health
    3 disorders: Anxiety, Depression, Insomnia
    4 genres: Metal, Pop, Classical, EDM
"""


def load_dataset():
    file = '../datasets/mxmh_survey_results.csv'
    data = pd.read_csv(file,
                       usecols=['Fav genre', 'Hours per day', 'Frequency [Classical]', 'Frequency [EDM]',
                                'Frequency [Metal]',
                                'Frequency [Pop]', 'Anxiety', 'Depression', 'Insomnia'])

    data = data[pd.notna(data['Fav genre'])]
    data = data[pd.notna(data['Hours per day'])]
    data = data[pd.notna(data['Frequency [Classical]'])]
    data = data[pd.notna(data['Frequency [EDM]'])]
    data = data[pd.notna(data['Frequency [Metal]'])]
    data = data[pd.notna(data['Frequency [Pop]'])]
    data = data[pd.notna(data['Anxiety'])]
    data = data[pd.notna(data['Depression'])]
    data = data[pd.notna(data['Insomnia'])]

    df = pd.DataFrame({'Fav genre': data['Fav genre'],
                       'Hours per day': data['Hours per day'],
                       'Frequency [Classical]': data['Frequency [Classical]'],
                       'Frequency [EDM]': data['Frequency [EDM]'],
                       'Frequency [Metal]': data['Frequency [Metal]'],
                       'Frequency [Pop]': data['Frequency [Pop]'],
                       'Anxiety': data['Anxiety'],
                       'Depression': data['Depression'],
                       'Insomnia': data['Insomnia']})
    display(df)

    return data
