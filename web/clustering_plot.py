from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from core import ROOT_PATH


def create_plot(sample_name, clustering_model, embedding_model):
    sample_embedding = embedding_model.generate(sample_name)
    cluster_label = clustering_model.predict(sample_embedding)

    df_sample = pd.Series({
        "name": sample_name,
        "embedding": sample_embedding,
        "cluster_label": cluster_label,
    }).to_frame().T

    data_path = Path("data/df_skills_embeddings.pkl")
    df = pd.read_pickle(ROOT_PATH / data_path)

    df_skills_embeddings = pd.concat([df_sample, df], axis=0)

    tsne = TSNE(n_components=2)
    embeddings = np.array(df_skills_embeddings["embedding"].tolist())
    embeddings_2d = tsne.fit_transform(embeddings)
    df_skills_embeddings.loc[:, ["x", "y"]] = embeddings_2d

    fig = px.scatter(
        df_skills_embeddings.iloc[1:],
        x="x",
        y="y",
        color="cluster_label",
        hover_name="name",
        title="Scatter Plot of Vacancy Embeddings Colored by Cluster Label",
        height=700
    )
    x, y = df_skills_embeddings.iloc[0][['x', 'y']]

    fig.add_scatter(
        x=[x],
        y=[y],
        marker=dict(color="red", size=20),
        name="Навыки кандидата"
    )

    fig.update_layout(hovermode="closest")
    return fig
