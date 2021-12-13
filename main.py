
#!/usr/bin/env python

# -*- coding: utf-8 -*-

#streamlit, openpyxl, folium, streamlit_folium, geopy, spacy, googletrans, gensim==3.8.3, pyLDAvis==2.1.2, matplotlib
#sklearn, termcolor, seaborn, plotly


#!pip install --upgrade spacy
#!python -m spacy download en_core_web_md
#!pip install googletrans
#!pip install --upgrade gensim==3.8.3
#!pip install pyLDAvis==2.1.2
#!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
#!unzip mallet-2.0.8.zip

import matplotlib.pyplot as plt

__author__ = ['Gonzalo Tomás', 'Ricardi', 'Javier Icaza', 'Pablo Yuste']

import streamlit as st

from to_excel import get_downlable_excel
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import numpy as np
#import back
import models
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import front
import api_ricardi
from nlp import get_plot_lda, get_plots
import pyLDAvis
import pandas as pd

list_cities_countries = [['Paris', 'France'],['London', 'UK',],['Bucharest', 'Romania'],
                         ['Sofia', 'Bulgary'],['Helsinki', 'Finland'],['Vienna', 'Austria'],['Minsk', 'Belarus'],
                         ['Brussels',  'Belgium'],['Sarajevo','Bosnia and Herzegovina'],['Sofia', 'Bulgaria'],
                         ['Zagreb', 'Croatia'],['Prague', 'Czechia'],['Copenhagen', 'Denmark'],['Tallinn', 'Estonia'],
                         ['Paris', 'France'],['Berlin', 'Germany'],['Athens', 'Greece'],['Reykjavik', 'Iceland'],
                         ['Dublin', 'Ireland'],['Rome', 'Italy'],['Riga', 'Latvia'],
                         ['Vilnius', 'Lithuania'],['Valletta', 'Malta'],['Chisinau', 'Moldova'],
                         ['Podgorica', 'Montenegro'],['Amsterdam', 'Netherlands'],['Skopje', 'North Macedonia'],
                         ['Oslo', 'Norway'],['Warsaw', 'Poland'], ['Lisbon', 'Portugal'],['Bucharest', 'Romania'],
                         ['Moscow', 'Russia'],['Belgrade', 'Serbia'],['Madrid', 'Spain'],
                         ['Bratislava', 'Slovakia'],['Ljubljana', 'Slovenia'],['Bern', 'Switzerland'],['Kiev', 'Ukraine']]

list_local = [['Paris', 'France'],['London', 'UK',],['Dublin','Ireland']]

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Analysis of Cities with Reddit",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


local = True #if we are working on local or not

def write():

    st.write('<style>'
             '.toolbar{visibility: hidden}'
             '.css-y37zgl{text-align: center}'
             'body{color:#6E6E64}'
             'footer{visibility: hidden}'
             '.Widget>label{color: #6E6E64}'
             '.stRadio .ay{color: #6E6E64}'
             '.sidebar .sidebar-content .sidebar-close .open-iconic{opacity: .5}'
             '.sidebar .sidebar-content{background-image: linear-gradient('
             '180deg,#DCDCDC,#f7f6f4}}'
             '''.reportview-container .main .block-container{{
            padding-top: 0rem;
        }}
        details[title="Click to view actions"]{{
                    display: none;
                }}'''
             '</style>',
             unsafe_allow_html=True)

    st.markdown('<h1><b>DYNAMIC DASHBOARD REDDIT CITIES</b></h1>', unsafe_allow_html=True);     st.write(" "); st.write(" "); st.write(" ")

    PAGES = {
        "MAP": None,
        "DATA":None,
        "NLP": None,
        "Machine Learning": None,
    }

    cols = st.columns(5)

    with cols[0]:
        selection = st.selectbox('Seleccionar el DASHBOARD', list(PAGES.keys()))

    local = st.radio('Are we working on local?', ['Yes', 'No'])
    if local == 'No':
        city_reddit = st.selectbox('REDDIT CITY:', [f'{elem[0]}, {elem[1]}' for elem in list_cities_countries])
    if local == 'Yes':
        city_reddit = st.selectbox('REDDIT CITY:', [f'{elem[0]}, {elem[1]}' for elem in list_local])

    print(city_reddit)

    if selection == 'MAP':
        # center on Liberty Bell
        m = folium.Map(location=[50, 15],
                       zoom_start=4, control_scale=True)
        # add marker for Liberty Bell
        # call to render Folium map in Streamlit

        geolocator = Nominatim(user_agent="my_user_agent")

        city_destination = city_reddit.split(',')[0]
        country_destination = city_reddit.split(',')[1]
        loc = geolocator.geocode(city_destination + ',' + country_destination)
        marker = folium.Marker(location=[loc.latitude, loc.longitude], tooltip='source',
                                      icon=folium.Icon(color='red', prefix='fa', icon='circle'),
                                      draggable=True)
        marker.add_to(m)

        #folium.GeoJson(geo_str, name='geojson').add_to(m)

        folium_static(m, width=1690, height=700)

    if selection == 'DATA':
        type_reddit = ['hot', 'top', 'rising', 'new']
        sel_type = st.selectbox('Select the type', type_reddit)
        city_reddit = city_reddit.split(',')[0]

        if local == 'Yes':
            if city_reddit == 'Paris':
                df = pd.read_csv('df_paris.csv', keep_default_na=False); df = df.where(pd.notnull(df), None)
            if city_reddit == 'Dublin':
                df = pd.read_csv('df_dublin.csv', keep_default_na=False); df = df.where(pd.notnull(df), None)
                df['Eng Title'] = df['title'].astype(str); df['Eng Comments'] = df['Comments'].astype(str)
            if city_reddit == 'London':
                df = pd.read_csv('df_london.csv', keep_default_na=False); df = df.where(pd.notnull(df), None)
                df['Eng Title'] = df['title'].astype(str); df['Eng Comments'] = df['Comments'].astype(str)
        elif local == 'No':
            df = api_ricardi.get_df(sel_type, city_reddit)

        df_style = front.style_df(df)

        st.markdown('')
        st.markdown(f'<b>Average polarity for {city_reddit}: {df["Polarity"].mean()}</b>', unsafe_allow_html=True)
        st.markdown(f'<b>Average subjectivity for {city_reddit}: {df["Subjectivity"].mean()}</b>',unsafe_allow_html=True)
        st.markdown('')
        st.dataframe(df_style)
        st.markdown('')

        cols[len(cols) - 1].markdown(get_excelbook(df_style, len(df)), unsafe_allow_html=True, )

        st.markdown('')
        model_names = ['linear','Random Forest', 'knn', 'bayesian', 'decision trees', 'svm',
                       'comparison of all these models']

        st.markdown('<h3><b>Select the ML model to use.</h3></b>', unsafe_allow_html=True)
        model_selected = st.radio('', model_names)

        if model_selected == 'comparison of all these models':
            x = ['Linear', 'RF', 'KNN', 'Bayes', 'DecisionTrees', 'SVM']
            mse_error = []; ma_error = []; r_squared = []
            y, test_target = models.use_model(df, model_selected, None)
            for elem in y:
                mse_error.append(np.sqrt(mean_squared_error(test_target, elem)))
                ma_error.append(mean_absolute_error(test_target, elem))
                r_squared.append(r2_score(test_target, elem))

            trace = go.Bar(x=x, y=mse_error, marker_color=['Gold', 'MediumTurquoise', 'LightGreen','violet', 'rosybrown', 'tomato'], showlegend=False)
            layout = go.Layout(title="Root Mean Squared error")
            data = [trace]
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig)

            trace = go.Bar(x=x, y=ma_error, marker_color=['Gold', 'MediumTurquoise', 'LightGreen','violet', 'rosybrown', 'tomato'], showlegend=False)
            layout = go.Layout(title="Mean Absolute Error")
            data = [trace]
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig)

            trace = go.Bar(x=x, y=r_squared, marker_color=['Gold', 'MediumTurquoise', 'LightGreen','violet', 'rosybrown', 'tomato'], showlegend=False)
            layout = go.Layout(title="R-Squared Coefficient")
            data = [trace]
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig)

        else:
            st.markdown('<h3><b>Please choose the error metric.</h3></b>', unsafe_allow_html=True)
            error_metric_1 = st.checkbox('MSE')
            error_metric_2 = st.checkbox('MAE', value=True)
            error_metric_3 = st.checkbox('R-Squared')
            error_metric = []
            if error_metric_1:
                error_metric.append('mse')
            if error_metric_2:
                error_metric.append('mae')
            if error_metric_3:
                error_metric.append('R-squared')

            final_error = models.use_model(df, model_selected, error_metric)
            st.markdown(''); st.markdown('')

            for elem in final_error:
                st.markdown(f'<h4><b>The {elem[1]} of the {model_selected} model is: {elem[0]}</h4></b>', unsafe_allow_html=True)

    if selection == 'NLP':
        type_reddit = ['hot', 'top', 'rising', 'new']
        sel_type = st.selectbox('Select the type', type_reddit)
        city_reddit = city_reddit.split(',')[0]

        if local == 'Yes':
            if city_reddit == 'Paris':
                df = pd.read_csv('df_paris.csv', keep_default_na=False); df = df.where(pd.notnull(df), None)
            if city_reddit == 'Dublin':
                df = pd.read_csv('df_dublin.csv', keep_default_na=False); df = df.where(pd.notnull(df), None)
                df['Eng Title'] = df['title'].astype(str); df['Eng Comments'] = df['Comments'].astype(str)
            if city_reddit == 'London':
                df = pd.read_csv('df_london.csv', keep_default_na=False); df = df.where(pd.notnull(df), None)
                df['Eng Title'] = df['title'].astype(str); df['Eng Comments'] = df['Comments'].astype(str)
        elif local == 'No':
            df = api_ricardi.get_df(sel_type, city_reddit)

        n = st.slider(label='Choose the number of topics', min_value=1, max_value=10, step=1)
        num, list_month, sea, mf_topics = api_ricardi.get_popular_values(df, n, city_reddit)
        st.markdown('<h3><b>Topic Retrieval!!</b></h3>', unsafe_allow_html=True)
        st.markdown(f'<b>The month with most popular posts is January with {list_month[2].values[0]}</b>', unsafe_allow_html=True)
        st.markdown(f'<b>The season with most popular posts is {sea[num.index(max(num))]} with {max(num)}</b>', unsafe_allow_html=True)
        st.markdown(f'<b>The {n} most common topics of the {city_reddit} subreddit are: {[elem for elem in mf_topics]}</b>', unsafe_allow_html=True)
        st.markdown('')

        st.markdown('<h4><b>LDA Visualization</h4></b>', unsafe_allow_html=True)
        vis, ldag = get_plot_lda(df, city_reddit)
        html_string = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(html_string, width=1300, height=800)

        st.markdown('')
        st.markdown('<h4><b>Relevant Topics Visualization</h4></b>', unsafe_allow_html=True)
        fig = get_plots(ldag)
        st.pyplot(fig)

    if selection == 'Machine Learning':
        if local == 'Yes':
            if city_reddit.split(',')[0] == 'Paris':
                df = pd.read_csv('df_paris.csv', keep_default_na=False);
                df = df.where(pd.notnull(df), None)

            if city_reddit.split(',')[0] == 'Dublin':
                df = pd.read_csv('df_dublin.csv', keep_default_na=False);
                df = df.where(pd.notnull(df), None)
                df['Eng Title'] = df['title'].astype(str)
                df['Eng Comments'] = df['Comments'].astype(str)

            if city_reddit.split(',')[0] == 'London':
                df = pd.read_csv('df_london.csv', keep_default_na=False);
                df = df.where(pd.notnull(df), None)
                df['Eng Title'] = df['title'].astype(str)
                df['Eng Comments'] = df['Comments'].astype(str)
        elif local == 'No':
            df = api_ricardi.get_df(sel_type, city_reddit)

        df["media"] = df['media'].apply(lambda x: isinstance(x, dict))
        df["selftext"] = df['selftext'].apply(lambda x: len(x) > 0)
        df["title_length"] = df["title"].apply(lambda x: len(x.split()))
        df = df.drop(["title"], axis=1)
        dummy = pd.get_dummies(df["media"])
        dummy = dummy.rename(columns={True: "Media"})
        dummy2 = pd.get_dummies(df["selftext"])
        dummy2 = dummy2.rename(columns={True: "Selftext"})
        dummy3 = pd.get_dummies(df["Verified user"])
        dummy3 = dummy3.rename(columns={True: "Verified user"})

        def int_or_none(x):
            if x == '':
                return 0
            else:
                return float(x)
        df['gilded'] = df['gilded'].apply(int_or_none)
        df['downs'] = df['downs'].apply(int_or_none)
        df['total_awards_received'] = df['total_awards_received'].apply(int_or_none)
        df['pinned'] = df['pinned'].apply(int_or_none)
        df['num_crossposts'] = df['num_crossposts'].apply(int_or_none)
        df["media"] = df['media'].apply(lambda x: isinstance(x, str))

        df["popularity"] = df['# comments'] * df["upvote ratio"]
        df = df.drop(["upvote ratio", "# comments"], axis=1)
        plt.hist(df["popularity"], alpha=0.5, bins=20, edgecolor="black")
        plt.title("Histogram of popularity")
        plt.ylabel("Frequency")
        plt.xlabel("variable")
        a, b = 15, 30,
        plt.axvline(a, color='g', linestyle='dashed', linewidth=3)
        plt.axvline(b, color='g', linestyle='dashed', linewidth=3)
        st.pyplot()


        # pacos tree.

        for i in df["popularity"]:
            if i < 15:
                df["popularity"].replace(i, 0, inplace=True)
            if 15 < i < 30:
                df["popularity"].replace(i, 1, inplace=True)
            if 30 < i:
                df["popularity"].replace(i, 2, inplace=True)

        from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
        from sklearn.model_selection import train_test_split  # Import train_test_split function
        from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
        # split dataset in features and target variable
        feature_cols = ["month", "gilded", "total_awards_received", "pinned", "num_crossposts", "User Karma",
                        "title_length", "media", "selftext", "Verified user"]
        X = df[feature_cols]  # Features
        y = df.popularity  # Target variable
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier()
        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = clf.predict(X_test)
        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        from sklearn.tree import export_graphviz
        from six import StringIO
        # from sklearn.externals.six import StringIO
        from IPython.display import Image
        import pydotplus
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        st.image(graph, use_column_width=True)

@st.cache(show_spinner=False)
def get_excelbook(df, len_df):
    return get_downlable_excel(df, len_df, 'DASHBOARD Excel', 'Descargar Dashboards como archivo Excel',)


if __name__ == "__main__":
    write()

