from typing import Union

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from core.embedding_models import EmbeddingModel, FastTextEmbeddingModel
from core.models.clustering_model import ClusteringModel, StackedModels
from web import ROOT_PATH
from web.clustering_plot import create_plot
from web.document_processor import PDFToText


@st.cache_resource(show_spinner="Загрузка модели fasttext...")
def load_model(
    name: str = "stacked_model_fasttext",
) -> tuple[StackedModels, Union[EmbeddingModel, FastTextEmbeddingModel], ClusteringModel, ClusteringModel]:
    embedding_mapping: dict[str, Union[EmbeddingModel, FastTextEmbeddingModel]] = {
        "LaBSE-en-ru": EmbeddingModel(),
        "fasttext": FastTextEmbeddingModel(),
    }

    models_mapping = {
        "stacked_model_fasttext": (
            StackedModels,
            ClusteringModel(),
            ClusteringModel(),
            embedding_mapping["fasttext"],
        )
    }
    model_object, clustering_model, clustering_position_model, embedding_model = models_mapping[name]

    clustering_model = clustering_model.load_model(ROOT_PATH / "checkpoints/clustering_model_mean_skills_embedding.pkl")
    clustering_position_model = clustering_model.load_model(
        ROOT_PATH / "checkpoints/clustering_model_position_embedding.pkl"
    )

    model = model_object().load_model(ROOT_PATH / "checkpoints/stacked_model_fasttext.pkl")
    return model, embedding_model, clustering_model, clustering_position_model


def run_server():
    st.set_page_config(page_title="Предсказание зарплаты", layout="wide", page_icon="🧊")

    if "model" not in st.session_state.keys():
        (
            st.session_state["model"],
            st.session_state["embedding_model"],
            st.session_state["clustering_model"],
            st.session_state["clustering_position_model"],
        ) = load_model()

    pretrained_model = st.session_state["model"]
    embedding_model = st.session_state["embedding_model"]
    clustering_model = st.session_state["clustering_model"]
    clustering_position_model = st.session_state["clustering_position_model"]

    st.image("https://upload.wikimedia.org/wikipedia/commons/b/be/Logo_blue%403x.png", width=100)

    st.title("Предсказание зарплаты по резюме")

    pdf_file = st.file_uploader(label="Загрузите резюме в формате .pdf", type="pdf")

    run_button = st.button("Запустить")

    if run_button:
        if pdf_file:
            col1, col2 = st.columns([1, 2])
            with col1:
                binary_data = pdf_file.getvalue()

                pdf_viewer(input=binary_data, width=700)
            with col2:
                with st.spinner("Строим эмбеддинги..."):
                    pdf_to_text = PDFToText(pdf_file)
                    extracted_text = pdf_to_text.extract_text()
                    extracted_entities = pdf_to_text.extract_entities(extracted_text)

                    position_str = extracted_entities["Желаемая должность"][0]
                    schedule_str = extracted_entities["График работы"][0]
                    education_str = extracted_entities["Образование"][0]
                    skills_str = extracted_entities["Навыки"][0]

                    concat_str = position_str + schedule_str + education_str + skills_str

                    skills_embedding = embedding_model.generate(skills_str)
                    predicted_cluster = clustering_model.predict(skills_embedding)

                    vacancy_embedding = embedding_model.generate(concat_str)
                    cluster_label = clustering_model.predict(skills_embedding)

                with st.spinner("Предсказываем зарплату..."):
                    st.subheader("Предсказанная зарплата")

                    result = int(pretrained_model.predict(vacancy_embedding, cluster_label=cluster_label) // 1000)
                    st.info(f"**Зарплата**: {result} тыс. руб.")

                with st.spinner("Обрабатываем документ..."):
                    st.subheader("Извлеченный текст")
                    with st.expander("Весь текст из файла"):
                        st.text_area(label="", value=extracted_text, height=400)

                    st.subheader("Извлеченные признаки")
                    for entity, values in extracted_entities.items():
                        if len(values[0]):
                            st.write(f"**{entity.capitalize()}**: {', '.join(values) if values else 'Не найдено'}")

                    if len(skills_str):
                        st.subheader("Диаграмма кластеров")
                        st.info(f"**Кластер**: {predicted_cluster}")
                        st.plotly_chart(
                            create_plot(
                                position_str,
                                clustering_position_model,
                                embedding_model,
                            ),
                            use_container_width=True,
                        )

        else:
            st.warning("Загрузите документ, чтобы продолжить")


if __name__ == "__main__":
    run_server()
