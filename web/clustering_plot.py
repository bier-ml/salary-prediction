from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from sklearn.manifold import TSNE

from core import ROOT_PATH
from core.embedding_models import EmbeddingModel, FastTextEmbeddingModel
from core.models.clustering_model import ClusteringModel


def create_plot(
    sample_name: str,
    clustering_model: ClusteringModel,
    embedding_model: Union[EmbeddingModel, FastTextEmbeddingModel],
) -> Figure:
    sample_embedding = embedding_model.generate(sample_name)
    cluster_label = clustering_model.predict(sample_embedding)

    df_sample = (
        pd.Series(
            {
                "custom_position": sample_name,
                "position_embedding": sample_embedding,
                "cluster_label_based_on_position_embedding": cluster_label,
            }
        )
        .to_frame()
        .T
    )

    data_path = Path("data/vacancies_with_skills_positions_embeddings_and_cluster_label_ft.pkl")
    df = pd.read_pickle(ROOT_PATH / data_path)

    df_skills_embeddings = pd.concat([df_sample, df], axis=0)

    tsne = TSNE(n_components=2)
    embeddings = np.array(df_skills_embeddings["position_embedding"].tolist())
    embeddings_2d = tsne.fit_transform(embeddings)
    df_skills_embeddings.loc[:, ["x", "y"]] = embeddings_2d

    fig = px.scatter(
        df_skills_embeddings.iloc[1:],
        x="x",
        y="y",
        color="cluster_label_based_on_position_embedding",
        hover_name="custom_position",
        # title="Scatter Plot of Vacancy Embeddings Colored by Cluster Label",
        height=500,
    )
    x, y = df_skills_embeddings.iloc[0][["x", "y"]]

    fig.add_scatter(x=[x], y=[y], marker={"color": "black", "size": 15, "symbol": "x"}, name="Кандидат")

    fig.update_layout(hovermode="closest")
    return fig
