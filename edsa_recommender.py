"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data handling dependencies
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    html_template = """
    <div style="background-color:black;padding:10px;border-radius:10px;margin:10px;">
    <h1 style="color:green;text-align:center;">EDSA Movie Recommendation Challenge</h1>
    <h2 style="color:white;text-align:center;">UNSUPERVISED LEARNING PREDICT - TEAM1</h2>
    </div>
    """

    title_template ="""
    <div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:20px;">
    <h1 style="color:white;text-align:center;">UNSUPERVISED LEARNING PREDICT</h1>
    <h2 style="color:white;text-align:center;">TEAM 1</h2>
    <h3 style="color:white;text-align:center;">Malibongwe Xulu</h3>
    <h3 style="color:white;text-align:center;">Nthabiseng Moela</h3>
    <h3 style="color:white;text-align:center;">Simangele Maphanga</h3>
    <h3 style="color:white;text-align:center;">Kgauhelo Mokgawa</h3>
    <h3 style="color:white;text-align:center;">Manko Mofokeng</h3>
    <h2 style="color:white;text-align:center;">14 December 2020</h2>
    </div>
    """

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home","Recommender System","About","Exploratory Data Analysis","Recommender Performance","Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    
    if page_selection == "Home":
        st.markdown(html_template.format('royalblue','white'), unsafe_allow_html=True)
        st.image('resources/imgs/image.png',use_column_width=True)
        #st.markdown(title_template, unsafe_allow_html=True)

    if page_selection == "About":
        #markup(page_selection)
        st.write("### Oveview: Flex your Unsupervised Learning skills to generate movie recommendations")
        
        # You can read a markdown file from supporting resources folder
        #if st.checkbox("Introduction"):
        st.subheader("Introduction to Unsupervised Learning Predict")
        st.write("""In today’s technology driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.""")
        st.write("""With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.""")
        st.write("""Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity.""")

        #if st.checkbox("Problem Statement"):
        st.subheader("Problem Statement of the Unsupervised Learning Predict")
        st.write("Build recommender systems to recommend a movie")

        #if st.checkbox("Data"):
        st.subheader("Data Overview")
        st.write("""This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!""")

        st.write("""For this Predict, we'll be using a special version of the MovieLens dataset which has enriched with additional data, and resampled for fair evaluation purposes.""")

        st.write("""### Source:""") 
        st.write("""The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB""")


        st.write("""### Supplied Files:
        genome_scores.csv - a score mapping the strength between movies and tag-related properties. Read more here

        genome_tags.csv - user assigned tags for genome-related scores

        imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.

        links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.

        sample_submission.csv - Sample of the submission format for the hackathon.

        tags.csv - User assigned for the movies within the dataset.

        test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.

        train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.""")

            # st.subheader("Raw Twitter data and label")
            # if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            #     st.write(raw[['sentiment', 'message']]) # will write the df to the page

    if page_selection == "Exploratory Data Analysis":
        st.title('Exploratory Data Science')

        if st.checkbox("ratings"):
            st.subheader("Movie ratings")
            st.image('resources/imgs/rating.png',use_column_width=True)

        # if st.checkbox("correlation"):
        #     st.subheader("Correlation between features")
        #     st.image('resources/imgs/correlation.png',use_column_width=True)
        
        if st.checkbox("genre wordcloud"):
            st.subheader("Top Genres")
            st.image('resources/imgs/genre_wordcloud.png',use_column_width=True)
        
        if st.checkbox("genres"):
            st.subheader("Top Genres")
            st.image('resources/imgs/top_genres.png',use_column_width=True)
        
        # if st.checkbox("movies released per year"):
        #     st.subheader("Movies released per year")
        #     st.image('resources/imgs/release_year.png',use_column_width=True)

        if st.checkbox("tags"):
            st.subheader("Top tags")
            st.image('resources/imgs/top_tags.png',use_column_width=True)

        if st.checkbox("cast"):
            st.subheader("Popular cast")
            st.image('resources/imgs/cast.png',use_column_width=True)

    # if page_selection == "Recommend a movie":
    #     st.title("Recommend a movie")
    #     sys = st.radio("Select an algorithm",
    #                    ('Content Based Filtering',
    #                     'Collaborative Based Filtering'))


    if page_selection == "Recommender Performance":
        st.title("Recommendation Performance Evaluation")
        st.write("RMSE of the recommendation models to show their performance")
        st.image('resources/imgs/performance_df.png',use_column_width=True)

    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
