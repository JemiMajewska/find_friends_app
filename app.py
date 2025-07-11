import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import setup, create_model, assign_model, plot_model, save_model, load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
from openai import OpenAI
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

env = dotenv_values(".env")
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']


def get_openai_client():
    key = st.session_state.get("openai_api_key")
    if not key:
        st.warning("Brakuje klucza OpenAI API.")
        st.stop()
    return OpenAI(api_key=key)


#
# MAIN
#
st.set_page_config(page_title="Welcome survey results", layout="centered")

# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()


# Uzyjemy nowych danych v2:
df = pd.read_csv('welcome_survey_simple_v2.csv', sep=';')
s = setup(df, session_id=123)
s.dataset_transformed.head()

#stworzymy nowy pipeline dla danych v2:
kmeans = create_model('kmeans', num_clusters=15)
df_with_clusters = assign_model(kmeans)
df_with_clusters["Cluster"].value_counts()

save_model(kmeans, 'welcome_survey_clustering_pipeline_v2', verbose=False)
kmeans_pipeline = load_model('welcome_survey_clustering_pipeline_v2')
df_with_clusters = predict_model(model=kmeans_pipeline, data=df)
df_with_clusters["Cluster"].value_counts()


# Stworzymy prompt, dla LLM-a w celu znalezienia odpowiednich nazw i opisów dla klastrów
cluster_descriptions = {}
for cluster_id in df_with_clusters['Cluster'].unique():
    cluster_df = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
    summary = ""
    for column in df_with_clusters:
        if column == 'Cluster':
            continue

        value_counts = cluster_df[column].value_counts()
        value_counts_str = ', '.join([f"{idx}: {cnt}" for idx, cnt in value_counts.items()])
        summary += f"{column} - {value_counts_str}\n"

    cluster_descriptions[cluster_id] = summary

# Pokaz nazwy klastrow wraz z opisami
prompt = "Użyliśmy algorytmu klastrowania."
for cluster_id, description in cluster_descriptions.items():
    prompt += f"\n\nKlaster {cluster_id}:\n{description}"

prompt += """
Wygeneruj najlepsze nazwy dla każdego z klasterów oraz ich opisy, biorąc pod uwagę te same zainteresowania

Użyj formatu JSON. Przykładowo:
{
    "Cluster 0": {
        "name": "Klaster 0",
        "description": "W tym klastrze znajdują się osoby, które..."
    },
    "Cluster 1": {
        "name": "Klaster 1",
        "description": "W tym klastrze znajdują się osoby, które..."
    }
}
"""
print(prompt)

# Preparing clusters and cluster descriptions

openai_client = get_openai_client()


response = openai_client.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ],
)


result = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
cluster_names_and_descriptions = json.loads(result)

with open("welcome_survey_cluster_names_and_descriptions_v2.json", "w") as f:
    f.write(json.dumps(cluster_names_and_descriptions))

with open("welcome_survey_cluster_names_and_descriptions_v2.json", "r") as f:
    print(json.loads(f.read()))


# Wracamy do zaladowania utworzonego modelu, nowego survey i cluster descriptionv2:    

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age", color="gender")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(same_cluster_df, x="edu_level", color="gender")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
    template="plotly_white"
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals", color="gender")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
    template="plotly_white"
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place", color="gender")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
    template="plotly_white"
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender", color="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
    template="plotly_white",
    showlegend=False,
)
st.plotly_chart(fig)