import streamlit as st
import pickle
import pandas as pd
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import os, json
import subprocess
from sklearn.decomposition import PCA

#betolti az adott modelt, ha nem talalja tajekoztat
def load_model(model_name):
    try:
        with open(f'{model_name}.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"{model_name} modell nem található.⚠️⚠️⚠️")
        return None
def load_test_data():
    with open('test_data.pkl', 'rb') as f:
        return pickle.load(f)
#model leirasokat betolti
with open('model_descriptions.json', 'r', encoding="utf-8") as desc_file:
    model_descriptions = json.load(desc_file)
st.title("🌟Érzelemfelismerő🌟")
# ha true akkor a kiertekelesi nezet jon elo es elrejti a szoveg osztalyozas inputot
if 'evaluation_mode' not in st.session_state:
    st.session_state.evaluation_mode = False
# ha az osztalyozas sikeres akkor igaz és megjeleníti a kiertekeles gombot
if 'classification_done' not in st.session_state:
    st.session_state.classification_done = False
selected_model_name = st.selectbox("🔍Válassz modellt", list(model_descriptions.keys()))
if selected_model_name != "KMeans_model":
    if not st.session_state.evaluation_mode:
        if selected_model_name in model_descriptions:
            with st.expander("📚Információ a modellről:", expanded=False):
                st.write(model_descriptions[selected_model_name])
        model = load_model(selected_model_name)
        X_test, y_test = load_test_data()
        if model:
            # tarolja osztalyozasi eredmenyt
            if 'sentiment_log' not in st.session_state:
                st.session_state.sentiment_log = []
            user_input = st.text_area("🚀Írd be a szöveget az érzelem osztályozásához!")
            use_multiline = st.checkbox("Többsoros szöveg", value=False)
            if st.button("Osztályozás"):
                if user_input.strip() == "":
                    st.warning("Kérlek, adj meg egy szöveget.")
                else:
                    if use_multiline:
                        user_input_lines = user_input.split('\n')
                        for line in user_input_lines:
                            if line.strip() != "":
                                sentiment = model.predict([line])[0]
                                st.success(f"Az érzelem: {sentiment}, ({line})")
                                st.session_state.sentiment_log.append((line, sentiment))
                    else:
                        sentiment = model.predict([user_input])[0]
                        st.success(f"Az érzelem: {sentiment}")
                        st.session_state.sentiment_log.append((user_input, sentiment))
                    st.session_state.classification_done = True
            #sidebar pie chart, az eddig beirt szovegek erzelmeinek eloszlasa
            if len(st.session_state.sentiment_log) > 0:
                sentiment_counts = pd.Series([s[1] for s in st.session_state.sentiment_log]).value_counts()
                fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values)])
                fig.update_layout(title='📊Érzelemeloszlás:', showlegend=True)
                st.sidebar.plotly_chart(fig, use_container_width=True)
                st.sidebar.subheader("Szövegek története és osztályozásuk:")
                for i, (input_text, sentiment) in enumerate(st.session_state.sentiment_log):
                    st.sidebar.write(f"{i+1}. Szöveg: '{input_text}' | Érzelem: {sentiment}")
            if st.session_state.classification_done:
                if st.button("Kiértékelés"):
                    st.session_state.evaluation_mode = True
                    st.session_state.classification_done = False
                    st.rerun()
                if "show_inputs" not in st.session_state:
                    st.session_state.show_inputs = False
                if "sentimentfix_text" not in st.session_state:
                    st.session_state.sentimentfix_text = ""
                if "inputfix_text" not in st.session_state:
                    st.session_state.inputfix_text = ""
                if st.button("Nem megelégedett a beosztással?"):
                    st.session_state.show_inputs = True
                if st.session_state.show_inputs:
                    st.write("Ez jelenleg a MultinomialNB modellt tanítja újra.")
                    st.write("Kérjük ne 1-1 szót írjon.")
                    st.write("Az AI nagyon sok külön üzenetet tanult már be. A legújabb információ neki ugyanannyi értékkel bír mint a legrégebbi,avagy ha megad egy érzelmet és egy szöveget közel biztos hogy nem fogja egyből megadni a várt érzelmet.")
                    st.session_state.sentimentfix_text = st.text_input(
                        "Add meg az érzelmet:",
                        value=st.session_state.sentimentfix_text,
                        key="sentiment_input"
                    )
                    st.session_state.inputfix_text = st.text_area(
                        "Add meg a szöveget:",
                        value=st.session_state.inputfix_text,
                        key="content_input"
                    )
                    if st.button("Fejlessze az AI-unkat."):
                        if not st.session_state.sentimentfix_text or not st.session_state.inputfix_text:
                            st.error("Mindkét mezőt ki kell tölteni az AI fejlesztéséhez! ⚠️")
                        else:
                            with st.spinner("Modell betanítása folyamatban... Kérlek, várj."):
                                try:
                                    import subprocess
                                    csv_file = 'tweet_emotions.csv'
                                    tweet_id = 47
                                    new_row = {
                                        'tweet_id': tweet_id,
                                        'sentiment': st.session_state.sentimentfix_text,
                                        'content': st.session_state.inputfix_text
                                    }
                                    try:
                                        df = pd.read_csv(csv_file)
                                    except FileNotFoundError:
                                        df = pd.DataFrame(columns=new_row.keys())
                                    new_row_df = pd.DataFrame([new_row])
                                    updated_df = pd.concat([new_row_df, df], ignore_index=True)
                                    updated_df.to_csv(csv_file, index=False)
                                    st.success(f"Új sor hozzáadva a {csv_file} fájlhoz.")
                                    result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
                                    if result.returncode == 0:
                                        st.success("Modell betanítása sikeres! ✅")
                                        st.text(result.stdout)
                                    else:
                                        st.error("Hiba történt a betanítás során! ⚠️")
                                        st.text(result.stderr)
                                except Exception as e:
                                    st.error(f"Hiba történt a betanítás során: {str(e)}")
    else:
        model = load_model(selected_model_name)
        X_test, y_test = load_test_data()
        if model:
            st.write("📈Modell kiértékelése")
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            if cm.size > 0:
                try: #konfuzios matrix, megmutatja, hogy melyik erzelem mivel lehet osszekeverve pl. worry&neutral. ha x: neutral y: worry = 532 az azt jelenti hogy 532 instance ami a neutral classhoz tartozna az lett rosszul osztalyozva es worry classhoz lett rendelve
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

                # pontossagi riport
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

        if st.button("Vissza"):
            st.session_state.evaluation_mode = False
            st.session_state.classification_done = False
            st.rerun()
else:
    if not st.session_state.evaluation_mode:
        if selected_model_name in model_descriptions:
            with st.expander("Információ a modellről:", expanded=False):
                st.write(model_descriptions[selected_model_name])

        model = load_model(selected_model_name)
        vectorizer = load_model("vectorizer")  # Load the vectorizer used for KMeans

        if model and vectorizer:
            if 'sentiment_log' not in st.session_state:
                st.session_state.sentiment_log = []

            user_input = st.text_area("Írd be a szöveget a klaszterezéshez!")
            use_multiline = st.checkbox("Többsoros szöveg", value=False)

            if st.button("Klaszterezés"):
                if user_input.strip() == "":
                    st.warning("Kérlek, adj meg egy szöveget.")
                else:
                    if use_multiline:
                        user_input_lines = user_input.split('\n')
                        for line in user_input_lines:
                            if line.strip() != "":
                                input_vectorized = vectorizer.transform([line])
                                cluster = model.predict(input_vectorized)[0]
                                st.success(f"A klaszter: {cluster}, ({line})")
                                st.session_state.sentiment_log.append((line, cluster))
                    else:
                        input_vectorized = vectorizer.transform([user_input])
                        cluster = model.predict(input_vectorized)[0]
                        st.success(f"A klaszter: {cluster}")
                        st.session_state.sentiment_log.append((user_input, cluster))

                    st.session_state.classification_done = True
            if len(st.session_state.sentiment_log) > 0:
                cluster_counts = pd.Series([s[1] for s in st.session_state.sentiment_log]).value_counts()
                fig = go.Figure(data=[go.Pie(labels=cluster_counts.index, values=cluster_counts.values)])
                fig.update_layout(title='Klasztereloszlás:', showlegend=True)
                st.sidebar.plotly_chart(fig, use_container_width=True)
                st.sidebar.subheader("Szövegek története és klaszterezésük:")
                for i, (input_text, cluster) in enumerate(st.session_state.sentiment_log):
                    st.sidebar.write(f"{i+1}. Szöveg: '{input_text}' | Klaszter: {cluster}")
            if st.session_state.classification_done:
                if st.button("Kiértékelés"):
                    st.session_state.evaluation_mode = True
                    st.session_state.classification_done = False
                    st.rerun()
    else:
        model = load_model(selected_model_name)
        vectorizer = load_model("vectorizer")
        if model and vectorizer:
            st.write("📈KMeans modell kiértékelése")
            try:
                data = pd.read_csv('tweet_emotions.csv')
                texts = data['content'].fillna('')
                X_test_vectorized = vectorizer.transform(texts)
                inertia = model.inertia_
                st.write(f"Model inercia: {inertia:.2f}")
                pca = PCA(n_components=2)
                X_embedded = pca.fit_transform(X_test_vectorized.toarray())
                y_clusters = model.predict(X_test_vectorized)
                fig = go.Figure()
                for cluster in range(model.n_clusters):
                    cluster_points = X_embedded[y_clusters == cluster]
                    fig.add_trace(go.Scatter(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode='markers',
                        name=f"Cluster {cluster}",
                        marker=dict(size=6),
                    ))
                fig.update_layout(
                    title="Klaszterek vizualizációja (PCA)",
                    xaxis_title="PCA komponens 1",
                    yaxis_title="PCA komponens 2",
                    showlegend=True,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Hiba történt a vizualizáció során: {e}")
        if st.button("Vissza"):
            st.session_state.evaluation_mode = False
            st.session_state.classification_done = False
            st.rerun()
