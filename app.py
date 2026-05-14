import streamlit as st
import joblib
import pandas as pd
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right, #141e30, #243b55);
    color:white;
}

/* Heading */
.main-title{
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#00f2fe;
    margin-bottom:5px;
}

.sub-title{
    text-align:center;
    color:white;
    font-size:18px;
    margin-bottom:30px;
}

/* Input Boxes */
.stNumberInput input{
    background-color: rgba(255,255,255,0.1);
    color:white;
    border-radius:10px;
}

/* Selectbox */
div[data-baseweb="select"]{
    background-color: rgba(255,255,255,0.1);
    border-radius:10px;
}

/* Button */
.stButton>button{
    width:100%;
    height:55px;
    border:none;
    border-radius:15px;
    background: linear-gradient(to right, #00f2fe, #4facfe);
    color:white;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background: linear-gradient(to right, #43e97b, #38f9d7);
    color:black;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("student_model.pkl")
columns = joblib.load("model_columns.pkl")

# =========================
# HEADING
# =========================
st.markdown("""
<div class='main-title'>🎓 Student Score Predictor</div>
<div class='sub-title'>
Predict student exam performance using Machine Learning
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT FIELDS
# =========================
hours = st.number_input("📚 Hours Studied", 0.0, 24.0)
attendance = st.number_input("🧾 Attendance (%)", 0.0, 100.0)
previous = st.number_input("📈 Previous Score", 0.0, 100.0)
sleep = st.number_input("😴 Sleep Hours", 0.0, 12.0)

motivation = st.selectbox(
    "🔥 Motivation Level",
    ["Low", "Medium", "High"]
)

teacher = st.selectbox(
    "👨‍🏫 Teacher Quality",
    ["Poor", "Average", "Good"]
)

school = st.selectbox(
    "🏫 School Type",
    ["Public", "Private"]
)

internet = st.selectbox(
    "🌐 Internet Access",
    ["Yes", "No"]
)

income = st.selectbox(
    "💰 Family Income",
    ["Low", "Medium", "High"]
)

parent = st.selectbox(
    "👨‍👩‍👦 Parental Involvement",
    ["Low", "Medium", "High"]
)

education = st.selectbox(
    "🎓 Parent Education",
    ["School", "College"]
)

peer = st.selectbox(
    "👥 Peer Influence",
    ["Negative", "Neutral", "Positive"]
)

resources = st.selectbox(
    "📖 Learning Resources",
    ["Low", "Medium", "High"]
)

activities = st.selectbox(
    "⚽ Extracurricular Activities",
    ["Yes", "No"]
)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🚀 Predict Score"):

    # Create Input Data
    data = {
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Previous_Scores": previous,
        "Sleep_Hours": sleep,

        "Motivation_Level": motivation,
        "Teacher_Quality": teacher,
        "School_Type": school,
        "Internet_Access": internet,
        "Family_Income": income,
        "Parental_Involvement": parent,
        "Parental_Education_Level": education,
        "Peer_Influence": peer,
        "Learning_Resources": resources,
        "Extracurricular_Activities": activities
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Encoding
    input_df = pd.get_dummies(input_df)

    # Match columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Loading Animation
    with st.spinner("Predicting Score..."):
        time.sleep(2)

        # Prediction
        prediction = model.predict(input_df)

    # Fix unrealistic values
    final_score = max(40, min(100, prediction[0]))
    final_score = int(round(final_score))

    # Result
    st.success(f"🎯 Predicted Exam Score: {final_score}")

    # Progress Bar
    st.progress(final_score)

    # Chart
    chart_data = pd.DataFrame({
        "Values": [
            hours,
            attendance,
            previous,
            sleep
        ]
    })

    st.line_chart(chart_data)
