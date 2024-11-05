import streamlit as st
import pickle
import pandas as pd
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import glob, os, json


def load_model():
    models = {}
    descriptions = {}
    
    with open('model_descriptions.json', 'r', encoding="utf-8") as desc_file:
        descriptions = json.load(desc_file)
    
    
    for model_file in glob.glob('*_model.pkl'):
        with open(model_file, 'rb') as f:
            model_name = os.path.splitext(model_file)[0]
            models[model_name] = pickle.load(f)
    return models, descriptions

def load_test_data():
    with open('test_data.pkl', 'rb') as f:
        return pickle.load(f)

loaded_models, model_descriptions = load_model()
X_test, y_test = load_test_data()

st.title("Érzelemfelismerő")

selected_model_name = st.selectbox("Válassz modellt", list(loaded_models.keys()))

if selected_model_name.lower() in model_descriptions:
    with st.expander("Információ a modellről:", expanded=False):
        st.write(model_descriptions[selected_model_name.lower()])


model = loaded_models[selected_model_name]



if 'classified' not in st.session_state:
    st.session_state.classified = False

user_input = st.text_area("Írd be a szöveget az érzelem osztályozásához!")

if st.button("Osztályozás"):
    if user_input.strip() == "":
        st.warning("Kérlek, adj meg egy szöveget.")
    else:
        sentiment = model.predict([user_input])[0]
        st.success(f"Az érzelem: {sentiment}")

        st.session_state.classified = True


if st.session_state.classified:
    if st.button("Kiértékelés"):
        st.write("Modell kiértékelése")

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

        if cm.size > 0:

            try: # konfuzios matrix, megmutatja, hogy melyik erzelem mivel lehet osszekeverve pl. worry&neutral. ha x: neutral y: worry = 532 az azt jelenti hogy 532 instance ami a neutral classhoz tartozna az lett rosszul osztalyozva es worry classhoz lett rendelve
                fig = ff.create_annotated_heatmap(
                    z=cm,
                    x=list(model.classes_),
                    y=list(model.classes_),
                    colorscale='Blues',
                    showscale=True
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Hiba történt a grafikon generálása során: {str(e)}")


            report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
            report_df = pd.DataFrame(report).transpose()


            fig2 = go.Figure(data=[
                go.Bar(name='F1 Score', x=report_df.index[:-3], y=report_df['f1-score'][:-3]),
                go.Bar(name='Precision', x=report_df.index[:-3], y=report_df['precision'][:-3]),
                go.Bar(name='Recall', x=report_df.index[:-3], y=report_df['recall'][:-3]),
            ])
            fig2.update_layout(barmode='group', title='Pontossági riport',
                               xaxis_title='Osztály', yaxis_title='Pontszám',
                               xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.error("A konfúziós mátrix nem generálódott megfelelően.")
