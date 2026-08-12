import streamlit as st
import pandas as pd
import folium
import requests
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="London Inequality Dashboard",
    page_icon="🇬🇧",
    layout="wide"
)

st.title("🇬🇧 London Urban Inequality Analyser")
st.markdown("Exploring socioeconomic deprivation across London's 32 boroughs using open government data and machine learning.")
st.markdown("---")

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("master_london.csv")

df = load_data()

# ── TRAIN MODEL ───────────────────────────────────────────
@st.cache_resource
def train_model(df):
    X = df[[
        "income_score", "employment_score", "health_score",
        "education_score", "crime_score_dep",
        "environment_score", "total_crime_count"
    ]]
    y = df["deprivation_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X, X_test, y_test

model, X, X_test, y_test = train_model(df)

# ── LOAD GEOJSON ──────────────────────────────────────────
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/radoi90/housequest-data/master/london_boroughs.geojson"
    return requests.get(url).json()

geojson_data = load_geojson()

# ── TOP METRICS ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

y_pred = model.predict(X_test)
r2     = r2_score(y_test, y_pred)

with col1:
    st.metric("Most Deprived Borough",
              df.loc[df["deprivation_score"].idxmax(), "borough"])
with col2:
    st.metric("Least Deprived Borough",
              df.loc[df["deprivation_score"].idxmin(), "borough"])
with col3:
    st.metric("London Average Score",
              f"{df['deprivation_score'].mean():.1f}")
with col4:
    st.metric("Model Accuracy R²", f"{r2:.1%}")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────
# Tabs keep it clean and simple — one thing at a time
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Map",
    "📊 Borough Ranking",
    "💡 What Drives Deprivation?",
    "🔍 Borough Deep Dive"
])

# ── TAB 1 — MAP ───────────────────────────────────────────
with tab1:
    st.subheader("Interactive Deprivation Map")
    st.caption("Darker colour = more deprived. Click any borough for details.")

    m = folium.Map(
        location=[51.5074, -0.1278],
        zoom_start=9,
        tiles="CartoDB positron"
    )

    folium.Choropleth(
        geo_data=geojson_data,
        data=df,
        columns=["borough", "deprivation_score"],
        key_on="feature.properties.name",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="Deprivation Score",
        highlight=True
    ).add_to(m)

    for _, row in df.iterrows():
        for feature in geojson_data["features"]:
            if feature["properties"]["name"] == row["borough"]:
                coords = feature["geometry"]["coordinates"]
                if feature["geometry"]["type"] == "Polygon":
                    points = coords[0]
                else:
                    points = max(coords, key=lambda x: len(x[0]))[0]
                lat = sum(p[1] for p in points) / len(points)
                lng = sum(p[0] for p in points) / len(points)

                popup_html = f"""
                <div style="font-family:Arial;width:200px;padding:8px">
                    <b style="color:#185FA5">{row['borough']}</b><hr>
                    <small>
                    Deprivation Score: <b>{row['deprivation_score']:.1f}</b><br>
                    Income: {row['income_score']:.3f}<br>
                    Employment: {row['employment_score']:.3f}<br>
                    Health: {row['health_score']:.3f}<br>
                    Total Crimes: {int(row['total_crime_count']):,}
                    </small>
                </div>
                """
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=5,
                    color="#185FA5",
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=f"{row['borough']}: {row['deprivation_score']:.1f}"
                ).add_to(m)
                break

    st_folium(m, width=900, height=550)

# ── TAB 2 — BOROUGH RANKING ───────────────────────────────
with tab2:
    st.subheader("London Borough Deprivation Ranking")
    st.caption("Red = most deprived · Green = least deprived · Blue = middle")

    df_sorted = df.sort_values("deprivation_score", ascending=True)
    colors = []
    for score in df_sorted["deprivation_score"]:
        if score > df["deprivation_score"].quantile(0.75):
            colors.append("#E74C3C")
        elif score < df["deprivation_score"].quantile(0.25):
            colors.append("#2ECC71")
        else:
            colors.append("#85B7EB")

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(df_sorted["borough"],
            df_sorted["deprivation_score"],
            color=colors)
    ax.set_xlabel("Deprivation Score (higher = more deprived)")
    ax.set_title("All 32 London Boroughs Ranked by Deprivation")
    plt.tight_layout()
    st.pyplot(fig)


# ── TAB 3 — SHAP ──────────────────────────────────────────
with tab3:
    st.subheader("What Drives Deprivation Most?")
    st.caption("SHAP analysis shows which factors the model relies on most")
    with st.spinner("Running SHAP analysis..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("Feature Importance — What Drives London Deprivation?")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
    st.markdown("**Key finding:** Employment deprivation is the strongest predictor of overall deprivation across London boroughs.")
# ── TAB 4 — BOROUGH DEEP DIVE ─────────────────────────────
with tab4:
    st.subheader("Borough Deep Dive")

    selected = st.selectbox(
        "Choose a borough",
        options=sorted(df["borough"].tolist())
    )

    row = df[df["borough"] == selected].squeeze()
    rank = int(df["deprivation_score"]
               .rank(ascending=False)[df["borough"] == selected]
               .values[0])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Deprivation Score", f"{row['deprivation_score']:.1f}")
    with m2:
        st.metric("London Rank", f"{rank} / 32")
    with m3:
        st.metric("Employment Score", f"{row['employment_score']:.3f}")
    with m4:
        st.metric("Total Crimes", f"{int(row['total_crime_count']):,}")

    st.markdown("**How this borough compares to London average:**")

    features = ["deprivation_score", "income_score",
                "employment_score", "health_score", "education_score"]
    compare = pd.DataFrame({
        "Feature": [f.replace("_score","").title() for f in features],
        selected:  [row[f] for f in features],
        "London Avg": [df[f].mean() for f in features]
    })

    fig3 = px.bar(
        compare.melt(id_vars="Feature",
                     var_name="Area",
                     value_name="Score"),
        x="Feature", y="Score",
        color="Area", barmode="group",
        color_discrete_map={
            selected: "#185FA5",
            "London Avg": "#95A5A6"
        },
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption("Data: English Indices of Deprivation 2019 (MHCLG) · MPS Crime Data · Built with Python, Scikit-learn, SHAP, Folium & Streamlit")
