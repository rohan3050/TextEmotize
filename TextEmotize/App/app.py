#core Packages
import streamlit as st
import altair as alt

#EDA Packages
import pandas as pd
import numpy as np

#Utilas
import joblib

pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03_june_2021.pkl","rb"))

#FXN
def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {"anger":"😡","disgust":"😒","fear":"😨","guilt":"😩","joy":"😂","sadness":"😢","shame":"😳"}

def main():
    st.title("TextEmotize")
    st.subheader(f'"Decode the Heart❤️ of Text💬💌"')
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    
    st.markdown("""<html><head></head><body>
                <div style="position: fixed; bottom: 0; right: 20px; font-size: 25px; color: white; font-weight: bold;">Team-"AI Artisans"</div><body></html>""", unsafe_allow_html=True)
    
    
    if choice == "Home":
        st.subheader(f"Home-Emotions In Text")
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')
           
            
        if submit_text:
            col1,col2 = st.columns(2)
            col3,col4 = st.columns(2)
            #col4 = st.column(1)
            
            #Apply fxn here
            predict = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)
            
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                
                st.success("Predictions")
                emoji_icon = emotions_emoji_dict[predict]
                st.write("{}:{}".format(predict, emoji_icon))
                st.write("Confidence : {}".format(np.max(probability)))
                

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability*100, columns=pipe_lr.classes_)
                
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]
                
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability', color='emotions')
                st.altair_chart(fig,use_container_width=True)
                
            with col3:
                st.success("probability in %")
                proba_df = proba_df.T*100
                st.write(proba_df.T)
            
            #with col4:
                #st.write(probability)
    elif choice == "Monitor":
        st.subheader("Monitor App")
        
    else:
        st.subheader("About")
        
if __name__ == '__main__':
    main()    
    
