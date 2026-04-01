import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ===============================
# LOAD MODELS
# ===============================
category_model = joblib.load("svm_email_classifier.pkl")
category_vectorizer = joblib.load("tfidf_vectorizer.pkl")

urgency_model = joblib.load("urgency_model.pkl")
urgency_vectorizer = joblib.load("urgency_vectorizer.pkl")

# ===============================
# RULE-BASED URGENCY
# ===============================
HIGH_PATTERNS = [
    r"urgent", r"asap", r"immediately",
    r"not working", r"error", r"failed"
]

MED_PATTERNS = [
    r"follow up", r"need help"
]

def rule_based_urgency(text):
    t = text.lower()
    for pat in HIGH_PATTERNS:
        if re.search(pat, t):
            return "High"
    for pat in MED_PATTERNS:
        if re.search(pat, t):
            return "Medium"
    return "Low"

def hybrid_urgency(text):
    vec = urgency_vectorizer.transform([text])
    ml_pred = urgency_model.predict(vec)[0]

    rule_pred = rule_based_urgency(text)

    if ml_pred == "Low" and rule_pred == "High":
        return "High"
    if ml_pred == "Low" and rule_pred == "Medium":
        return "Medium"

    return ml_pred

# ===============================
# PREDICT FUNCTION
# ===============================
def predict_email(text):
    vec = category_vectorizer.transform([text])
    category = category_model.predict(vec)[0]
    urgency = hybrid_urgency(text)
    return category, urgency

# ===============================
# UI DESIGN
# ===============================
st.title("📧 AI Smart Email Classifier")

menu = st.sidebar.selectbox("Menu", ["Predict Email", "Dashboard"])

# ===============================
# 🔹 PAGE 1: PREDICTION
# ===============================
if menu == "Predict Email":
    st.subheader("Enter Email Text")

    email_input = st.text_area("Type email here...")

    if st.button("Classify"):
        category, urgency = predict_email(email_input)

        st.success(f"Category: {category}")
        st.warning(f"Urgency: {urgency}")

# ===============================
# 🔹 PAGE 2: DASHBOARD
# ===============================
elif menu == "Dashboard":

    df = pd.read_csv("clean_email.csv")

    st.subheader(" Email Analytics")

    # Filters
    category_filter = st.selectbox("Filter by Category", ["All"] + list(df["Category"].unique()))
    urgency_filter = st.selectbox("Filter by Urgency", ["All"] + list(df["Urgency"].unique()))

    filtered_df = df.copy()

    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["Category"] == category_filter]

    if urgency_filter != "All":
        filtered_df = filtered_df[filtered_df["Urgency"] == urgency_filter]

    st.write("Filtered Data:", filtered_df.shape)

    # Category Chart
    st.subheader("Category Distribution")
    fig1, ax1 = plt.subplots()
    filtered_df["Category"].value_counts().plot(kind="bar", ax=ax1)
    st.pyplot(fig1)

    # Urgency Chart
    st.subheader("Urgency Distribution")
    fig2, ax2 = plt.subplots()
    filtered_df["Urgency"].value_counts().plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

    # Top categories
    st.subheader("Top Categories")
    st.write(filtered_df["Category"].value_counts().head())
