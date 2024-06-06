import streamlit as st
from pypdf import PdfReader

from core.embedding_models import EmbeddingModel, FastTextEmbeddingModel
from core.models.clustering_model import StackedModels, ClusteringModel
from web import ROOT_PATH


class MockNERModel:
    def __init__(self):
        self.entities = {
            "job_name": ["Software Engineer", "Data Scientist"],
            "schedule": ["Full-time", "Part-time"],
            "city": ["San Francisco", "New York"],
        }

    def predict(self, text):
        extracted_entities = {"job_name": [], "schedule": [], "city": []}
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


@st.cache_resource()
def load_model(
    name: str = "stacked_model_fasttext",
) -> tuple[StackedModels, EmbeddingModel | FastTextEmbeddingModel]:
    embedding_mapping = {
        "LaBSE-en-ru": EmbeddingModel(),
        "fasttext": FastTextEmbeddingModel(),
    }

    models_mapping = {
        "stacked_model_fasttext": (
            StackedModels,
            ClusteringModel().load_model(
                ROOT_PATH / "checkpoints/clustering_model_fasttext.pkl"
            ),
            embedding_mapping["fasttext"],
        )
    }
    model_object, clustering_model, embedding_model = models_mapping[name]

    model = model_object(clustering_model=clustering_model).load_model(
        ROOT_PATH / "checkpoints/stacked_model_fasttext.pkl"
    )
    return model, embedding_model


AVAILABLE_SCHEDULE: tuple = (
    "полный рабочий день",
    "частичная занятость",
    "удаленная работа",
    "сменный график",
    "свободный график",
    "вахта",
)

if "model" not in st.session_state.keys():
    st.session_state["model"], st.session_state["embedding_model"] = load_model()


def run_server():
    # st.set_page_config(layout="wide")

    pretrained_model = st.session_state["model"]
    embedding_model = st.session_state["embedding_model"]

    st.markdown(
        """
        <a href="https://streamlit.io">
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/be/Logo_blue%403x.png" style="width:100px;">
        </a>
        """,
        unsafe_allow_html=True,
    )

    st.title("Предсказание зарплаты по резюме")

    pdf_file = st.file_uploader(label="Загрузите резюме в формате .pdf", type="pdf")

    run_button = st.button("Запустить")
    result_placeholder = st.empty()

    model = "test"
    source = "test"
    if run_button:
        if pdf_file:  # job_name and model and schedule and city:
            pdf_to_text = PDFToText(pdf_file)
            extracted_text = pdf_to_text.extract_text()
            st.subheader("Extracted Text")
            st.text_area(label="", value=extracted_text, height=400)

            extracted_entities = pdf_to_text.extract_entities(extracted_text)
            st.subheader("Extracted Entities")
            for entity, values in extracted_entities.items():
                st.write(
                    f"**{entity.capitalize()}**: {', '.join(values) if values else 'None found'}"
                )

            result_placeholder.text("Processing...")
            st.subheader("Predicted Salary")

            source_embedding = embedding_model.generate(source)
            result = pretrained_model.predict(source_embedding)

            result_placeholder.text("0")
            result_placeholder.write(f"Predicted salary is: {result}")
        else:
            st.warning("Загрузите документ, чтобы продолжить")


if __name__ == "__main__":
    run_server()
