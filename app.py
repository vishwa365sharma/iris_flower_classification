
import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸", layout="centered")

st.markdown("""
<style>
.stApp{
background: linear-gradient(138deg,#0f172a,#1e3a8a,#06b6d4);
}
.block-container{padding-top:2rem;}
.card{
background:rgba(255,255,255,.12);
padding:25px;border-radius:18px;
backdrop-filter:blur(10px);
}
h1,p,label{color:white!important;}
.stButton>button{
width:100%;height:50px;border-radius:10px;
background:#16a34a;color:white;font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    model=pickle.load(open("best_model.pkl","rb"))
    try:
        scaler=pickle.load(open("scaler.pkl","rb"))
    except:
        scaler=None
    return model,scaler

model,scaler=load()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements and click Predict.")

sl=st.number_input("Sepal Length (cm)", min_value=0.0, value=0.0)
sw=st.number_input("Sepal Width (cm)", min_value=0.0, value=0.0)
pl=st.number_input("Petal Length (cm)", min_value=0.0, value=0.0)
pw=st.number_input("Petal Width (cm)", min_value=0.0, value=0.0)

if st.button("🚀 Predict Flower"):
    x=np.array([[sl,sw,pl,pw]])
    if scaler is not None:
        x=scaler.transform(x)
    pred=model.predict(x)[0]
    names={0:"🌼 Iris Setosa",1:"🌷 Iris Versicolor",2:"🌹 Iris Virginica"}
    st.success(f"Prediction: {names[int(pred)]}")
    if hasattr(model,"predict_proba"):
        prob=model.predict_proba(x)[0]
        st.progress(float(np.max(prob)))
        st.write(f"Confidence: {np.max(prob)*100:.2f}%")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<center style='color:white'>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
