import streamlit as st
import pickle
import pandas as pd
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go


def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        return pickle.load(f)

def load_test_data():
    with open('test_data.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
X_test, y_test = load_test_data()

st.title("Érzelemfelismerő")
st.write("Írd be a szöveget az érzelem osztályozásához!")


if 'classified' not in st.session_state:
    st.session_state.classified = False

user_input = st.text_area("Írd be a szöveget:")

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

            try:
                fig = ff.create_annotated_heatmap(
                    z=cm,
                    x=model.classes_,
                    y=model.classes_,
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
