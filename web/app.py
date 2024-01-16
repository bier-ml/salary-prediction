import pandas as pd
import streamlit as st

from models.model_loader import knn_model


def run_server():
    st.set_page_config(layout="wide")

    st.title("Salary prediction app")
    job_name = st.text_input("Enter job title")
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox(label="Select model", options=["knn"])

    with col2:
        schedule = st.selectbox(
            label="Select schedule",
            options=[
                "полный рабочий день",
                "частичная занятость",
                "удаленная работа",
                "сменный график",
                "свободный график",
                "вахта",
            ],
        )

    with col3:
        city = st.selectbox(label="Select city_id", options=[1, 57, 2, 102, 174])

    data = pd.DataFrame(
        {"custom_position": [job_name], "schedule": [schedule], "city_id": [city]}
    )

    run_button = st.button("Run prediction")
    result_placeholder = st.empty()
    result = ""
    if run_button:
        if job_name and model and schedule and city:
            result_placeholder.text("Processing...")
            if model == "knn":
                if knn_model:
                    result = knn_model.predict(data)
                else:
                    st.error("Loading model error")

            result_placeholder.text(f"Predicted salary is: {result[0]}")
        else:
            st.error("Fill in required fields")


if __name__ == "__main__":
    run_server()
