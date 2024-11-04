import streamlit as st
import pickle

# Modell betolt
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Streamlit
st.title("Érzelemfelismerő")
st.write("Írd be a szöveget az érzelem osztályozásához!")


user_input = st.text_area("Írd be a szöveget:")


if st.button("Osztályozás"):
    if user_input.strip() == "":
        st.warning("Kérlek, adj meg egy szöveget.")
    else:
        sentiment = model.predict([user_input])[0]
        st.success(f"Az érzelem: {sentiment}")


#futtatashoz ALT+f12 es streamlit run GUI.py
