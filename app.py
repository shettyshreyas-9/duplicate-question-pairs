import streamlit as st
import helper
import pickle

# lr= pickle.load(open('lr.pkl','rb'))

st.header('Duplicate Question Pairs')

q1= st.text_input('Enter question 1')
q2= st.text_input('Enter question 2')


if st.button('Find'):
    query= helper.query_point_creator(q1,q1)

    st.header(query)