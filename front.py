
# front.py

# In this file, we will use the platform streamlit.io to create an interactive dashboard.

import streamlit as st
import pandas as pd



def style_df(df):

    def color_negative(s):
        return 'color: red;' if s == 0 else 'color:black'

    df_style = df.style.applymap(color_negative)
    df_style.format({'year': lambda x: int(x),
                     'month': lambda x: int(x),
                     'day': lambda x: int(x),
                     'score': lambda x: int(x),
                     '# comments': lambda x: int(x),
                     'media only': lambda x: int(x),
                     'video': lambda x: int(x),
                     #'clicked': lambda x: int(x),
                     #'downs': lambda x: int(x),
                     #'total_awards_received': lambda x: int(x),
                     #'ups': lambda x: int(x),
                     #'pinned': lambda x: int(x),
                     #'num_crossposts': lambda x: int(x),
                     }).applymap(lambda x: 'font-weight: bold',
                                 subset=pd.IndexSlice[:, ['Topic']])

    return df_style