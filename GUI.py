import streamlit as st
import pickle
import pandas as pd
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import os, json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#betolti az adott modelt, ha nem talalja tajekoztat
def load_model(model_name):
    try:
        with open(f'{model_name}.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"{model_name} modell nem található.")
        return None

def load_test_data():
    with open('test_data.pkl', 'rb') as f:
        return pickle.load(f)

#model leirasokat betolti
with open('model_descriptions.json', 'r', encoding="utf-8") as desc_file:
    model_descriptions = json.load(desc_file)

st.title("Érzelemfelismerő")


# ha true akkor a kiertekelesi nezet jon elo es elrejti a szoveg osztalyozas inputot
if 'evaluation_mode' not in st.session_state:
    st.session_state.evaluation_mode = False

# ha az osztalyozas sikeres akkor igaz és megjeleníti a kiertekeles gombot
if 'classification_done' not in st.session_state:
    st.session_state.classification_done = False

selected_model_name = st.selectbox("Válassz modellt", list(model_descriptions.keys()))
if selected_model_name != "KMeans_model":
    if not st.session_state.evaluation_mode:
        if selected_model_name in model_descriptions:
            with st.expander("Információ a modellről:", expanded=False):
                st.write(model_descriptions[selected_model_name])

        model = load_model(selected_model_name)
        X_test, y_test = load_test_data()

        if model:
            # tarolja osztalyozasi eredmenyt
            if 'sentiment_log' not in st.session_state:
                st.session_state.sentiment_log = []

            user_input = st.text_area("Írd be a szöveget az érzelem osztályozásához!")
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

            #sidebar piec hart, az eddig beirt szovegek erzelmeinek eloszlasa
            if len(st.session_state.sentiment_log) > 0:
                sentiment_counts = pd.Series([s[1] for s in st.session_state.sentiment_log]).value_counts()
                fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values)])
                fig.update_layout(title='Érzelemeloszlás:', showlegend=True)
                st.sidebar.plotly_chart(fig, use_container_width=True)
                st.sidebar.subheader("Szövegek története és osztályozásuk:")
                for i, (input_text, sentiment) in enumerate(st.session_state.sentiment_log):
                    st.sidebar.write(f"{i+1}. Szöveg: '{input_text}' | Érzelem: {sentiment}")
            if st.session_state.classification_done:
                if st.button("Kiértékelés"):
                    st.session_state.evaluation_mode = True
                    st.session_state.classification_done = False
                    st.rerun()

    else:
        model = load_model(selected_model_name)
        X_test, y_test = load_test_data()

        if model:
            st.write("Modell kiértékelése")
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
                                st.success(f"A klaszter: {cluster}, ({line})")  # Display the cluster number for multi-line input
                                st.session_state.sentiment_log.append((line, cluster))  # Save the cluster number, not the sentiment label
                    else:
                        input_vectorized = vectorizer.transform([user_input])
                        cluster = model.predict(input_vectorized)[0]
                        st.success(f"A klaszter: {cluster}")  # Display the cluster number for single-line input
                        st.session_state.sentiment_log.append((user_input, cluster))  # Save the cluster number, not the sentiment label

                    st.session_state.classification_done = True

            # Sidebar pie chart for cluster distribution
            if len(st.session_state.sentiment_log) > 0:
                cluster_counts = pd.Series([s[1] for s in st.session_state.sentiment_log]).value_counts()
                fig = go.Figure(data=[go.Pie(labels=cluster_counts.index, values=cluster_counts.values)])
                fig.update_layout(title='Klasztereloszlás:', showlegend=True)
                st.sidebar.plotly_chart(fig, use_container_width=True)
                st.sidebar.subheader("Szövegek története és klaszterezésük:")
                for i, (input_text, cluster) in enumerate(st.session_state.sentiment_log):
                    st.sidebar.write(f"{i+1}. Szöveg: '{input_text}' | Klaszter: {cluster}")

            # Kiértékelés button
            if st.session_state.classification_done:
                if st.button("Kiértékelés"):
                    st.session_state.evaluation_mode = True
                    st.session_state.classification_done = False
                    st.rerun()

    else:
        # Evaluation logic for KMeans
        model = load_model(selected_model_name)
        vectorizer = load_model("vectorizer")  # Load the vectorizer used for KMeans

        if model and vectorizer:
            st.write("KMeans modell kiértékelése")

            try:
                # Load test data (same dataset with tweet emotions)
                data = pd.read_csv('tweet_emotions.csv')  # Load the same dataset
                texts = data['content'].fillna('')
                X_test_vectorized = vectorizer.transform(texts)

                # Get inertia (how well the clusters are defined)
                inertia = model.inertia_
                st.write(f"Model inercia: {inertia:.2f}")

                # Perform PCA for dimensionality reduction (2D visualization)
                pca = PCA(n_components=2)
                X_embedded = pca.fit_transform(X_test_vectorized.toarray())

                # Get the cluster predictions
                y_clusters = model.predict(X_test_vectorized)

                # Create a scatter plot (2D) with PCA components
                fig = go.Figure()

                # Iterate through the clusters and plot each one
                for cluster in range(model.n_clusters):
                    # Get the points belonging to this cluster
                    cluster_points = X_embedded[y_clusters == cluster]
                    fig.add_trace(go.Scatter(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode='markers',
                        name=f"Cluster {cluster}",
                        marker=dict(size=6),
                    ))

                # Update the plot layout
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
# cluster tesztelés:I absolutely love the smell of fresh flowers.
# I feel so guilty for forgetting her birthday.
# The movie was so disappointing; I expected better.
# I’m thrilled to be going on vacation next week.
# I feel a deep sense of nostalgia when I visit my childhood home.
# The thought of losing my job fills me with anxiety.
# I’m so proud of how much I’ve achieved this year.
# I can’t believe I embarrassed myself in front of everyone.
# The relief I felt when the test was over was immense.
# I feel a lot of jealousy when I see my friends getting promotions.
# Grief is overwhelming when you lose someone close.
# I was ecstatic to hear about the surprise party they planned for me.
# I regret not taking the opportunity when it was offered.
# I feel indifferent about the new movie that just came out.
# I love watching the sunset; it brings me peace.
# I’ve been feeling so bored lately with nothing to do.
# The surprise gift made me incredibly happy.
# I can't stop laughing at the silly joke he made.
# I hate the way she talks behind my back.
# I was shocked when I heard the news about the accident.