import streamlit as st

from core.embedding_models import EmbeddingModel
from core.models import CatboostRegressionModel
from core.models.base_model import BaseMatchingModel
from web import ROOT_PATH


def load_model(emb_model, name: str = "catboost_rubert-tiny2.pkl") -> BaseMatchingModel:
    cb_model = CatboostRegressionModel(embedding_model=emb_model)
    cb_model.load_model(ROOT_PATH / f"data/{name}")
    return cb_model


embedding_model = EmbeddingModel()
trained_model = load_model(embedding_model)


AVAILABLE_SCHEDULE: tuple = (
    "полный рабочий день",
    "частичная занятость",
    "удаленная работа",
    "сменный график",
    "свободный график",
    "вахта",
)

AVAILABLE_MODELS: tuple = tuple(
    [el.name for el in list((ROOT_PATH / "data/").glob("*.pkl"))]
)


def run_server():
    st.set_page_config(layout="wide")

    st.title("Salary prediction app")
    job_name = st.text_input("Enter job title", value="Программист")
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox(label="Select model", options=AVAILABLE_MODELS)

    with col2:
        schedule = st.selectbox(
            label="Select schedule",
            options=AVAILABLE_SCHEDULE,
        )

    with col3:
        city = st.selectbox(label="Select city_id", options=[1, 57, 2, 102, 174])

    source = "\n".join([job_name, schedule, str(city)])
    run_button = st.button("Run prediction")
    result_placeholder = st.empty()

    if run_button:
        if job_name and model and schedule and city:
            try:
                trained_model = load_model(embedding_model, model)
            except:
                st.error("Loading model error")

            result_placeholder.text("Processing...")
            result = trained_model.predict(source)
            result_placeholder.text(f"Predicted salary is: {result}")
        else:
            st.error("Fill in required fields")


if __name__ == "__main__":
    run_server()
