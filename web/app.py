import streamlit as st

from core.embedding_models import EmbeddingModel
from core.models import CatboostRegressionModel, LinearRegressionModel
from core.models.base_model import BaseMatchingModel
from web import ROOT_PATH
from pypdf import PdfReader

embedding_mapping = {
    "rubert-tiny2": EmbeddingModel(),
    "fasttext": EmbeddingModel(),
}


class MockNERModel:
    def __init__(self):
        self.entities = {
            "job_name": ["Software Engineer", "Data Scientist"],
            "schedule": ["Full-time", "Part-time"],
            "city": ["San Francisco", "New York"]
        }

    def predict(self, text):
        extracted_entities = {
            "job_name": [],
            "schedule": [],
            "city": []
        }
        for entity, examples in self.entities.items():
            for example in examples:
                if example in text:
                    extracted_entities[entity].append(example)
        return extracted_entities


class PDFToText:
    def __init__(self, file):
        self.file = file
        self.reader = PdfReader(self.file)

    def extract_text(self):
        text = ""
        for page in self.reader.pages:
            text += page.extract_text()
        return text

    def extract_entities(self, text):
        ner_model = MockNERModel()
        return ner_model.predict(text)


def load_model(name: str = "catboost_rubert-tiny2.pkl") -> BaseMatchingModel:
    model_mapping = {
        "catboost": CatboostRegressionModel,
        "linreg": LinearRegressionModel,
    }
    model_type = name.split("_")[0]
    emb_model = embedding_mapping[name.split(".")[0].split("_")[1]]

    model = model_mapping[model_type](embedding_model=emb_model)
    model.load_model(ROOT_PATH / f"data/{name}")

    return model


AVAILABLE_SCHEDULE: tuple = (
    "полный рабочий день",
    "частичная занятость",
    "удаленная работа",
    "сменный график",
    "свободный график",
    "вахта",
)

AVAILABLE_MODELS: tuple = tuple([el.name for el in list((ROOT_PATH / "data/").glob("*.pkl"))])


def run_server():
    st.set_page_config(layout="wide")

    st.markdown(
        """
        <a href="https://streamlit.io">
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/be/Logo_blue%403x.png" style="width:100px;">
        </a>
        """,
        unsafe_allow_html=True
    )

    st.title("Предсказание зарплаты по резюме")

    pdf_file = st.file_uploader(label="Загрузите резюме в формате .pdf", type='pdf')

    run_button = st.button("Запустить")
    result_placeholder = st.empty()

    model = 'test'
    source = 'test'
    if run_button:
        if pdf_file:  # job_name and model and schedule and city:
            pdf_to_text = PDFToText(pdf_file)
            extracted_text = pdf_to_text.extract_text()
            st.subheader("Extracted Text")
            st.text_area(label="", value=extracted_text, height=400)

            # try:
            #     trained_model = load_model(model)
            # except Exception as e:
            #     st.error(f"Loading model error: {str(e)}")

            extracted_entities = pdf_to_text.extract_entities(extracted_text)
            st.subheader("Extracted Entities")
            for entity, values in extracted_entities.items():
                st.write(f"**{entity.capitalize()}**: {', '.join(values) if values else 'None found'}")

            result_placeholder.text("Processing...")
            st.subheader("Predicted Salary")
            result = 0 #trained_model.predict(source)
            result_placeholder.text("0")
            result_placeholder.write(f"Predicted salary is: {result}")
        else:
            st.warning("Загрузите документ, чтобы продолжить")


if __name__ == "__main__":
    run_server()
