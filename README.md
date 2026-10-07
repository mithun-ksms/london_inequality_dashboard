# london_inequality_dashboard
# AI-Driven Analysis of Deprivation Across London Boroughs

## Overview

London is one of the wealthiest cities in the world, but significant inequality exists across its 32 boroughs.

This project uses **machine learning, socioeconomic data, crime data, and explainable AI** to investigate patterns of deprivation across London boroughs.

Rather than looking at individual deprivation indicators separately, the project combines multiple factors to understand which variables are most strongly associated with overall deprivation and whether machine learning can accurately predict deprivation scores.

The analysis also presents the results through an **interactive geospatial dashboard**, making the findings easier to explore and understand.

---

## Research Questions

The project focuses on three main research questions:

1. **Which socioeconomic factors most strongly predict deprivation scores across London boroughs?**
2. **Can a machine learning model accurately predict overall deprivation from a combination of socioeconomic indicators?**
3. **What spatial patterns emerge when multiple deprivation indicators are mapped at borough level?**

---

## Data Sources

The project combines publicly available datasets:

### 1. English Indices of Deprivation 2019

The **English Indices of Deprivation 2019 (IoD2019)** provides deprivation measures across seven domains:

* Income
* Employment
* Education
* Health
* Crime
* Housing
* Living environment

The original dataset operates at **Lower Super Output Area (LSOA)** level and was aggregated to London's 32 boroughs for this project.

### 2. Metropolitan Police Crime Data

Metropolitan Police borough-level crime data was used to calculate total recorded crime across London's boroughs.

The dataset covers **July 2024 to June 2026**, with crime records broken down by category and month.

### 3. London Borough Boundary Data

London borough boundary data was used to create the geospatial visualisations and interactive maps.

---

## Methodology

### Data Preprocessing

The IoD2019 dataset originally contained approximately **32,844 LSOA-level rows**.

The data was:

1. Filtered to London.
2. Aggregated from LSOA level to borough level.
3. Mean values were calculated for deprivation indicators.
4. Crime records were aggregated across the 24-month period.
5. Records with an `Unknown` location were removed.
6. The datasets were merged using borough name.
7. A final dataset containing **32 London boroughs** was produced.

The resulting master dataset contained **32 rows and 9 features**.

---

## Exploratory Data Analysis

A correlation heatmap was used to investigate relationships between the variables.

Some of the strongest relationships included:

| Relationship                      | Correlation |
| --------------------------------- | ----------: |
| Income ↔ Employment               |        0.97 |
| Overall Deprivation ↔ Income      |        0.98 |
| Overall Deprivation ↔ Employment  |        0.96 |
| Overall Deprivation ↔ Crime Count |        0.40 |

The relatively weak relationship between total crime count and deprivation was particularly interesting.

For example, **Westminster has very high recorded crime levels**, but this does not make it one of London's most deprived boroughs. High visitor and worker footfall may contribute to this difference.

---

## Machine Learning Model

### Random Forest Regressor

A **Random Forest Regressor** was selected because:

* The dataset contains only 32 boroughs.
* Random Forest performs well on structured datasets.
* It can model non-linear relationships.
* It can handle interactions between multiple features.
* It works well with SHAP for model interpretation.

The model used:

* **80/20 train-test split**
* **25 boroughs for training**
* **7 boroughs for testing**
* **100 estimators**
* Fixed random state for reproducibility

---

## Model Performance

The Random Forest achieved:

| Metric |    Result |
| ------ | --------: |
| R²     | **0.897** |
| RMSE   | **1.404** |

The R² score indicates that the model explained approximately **89.7% of the variance** in deprivation scores on the held-out test set.

However, because the dataset contains only 32 boroughs, this result should be interpreted cautiously. A high score on such a small dataset does not automatically mean the model would generalise well to other datasets or geographic areas.

---

## Explainable AI — SHAP

To understand what influenced the model's predictions, the project uses **SHAP (SHapley Additive exPlanations)** with a TreeExplainer.

SHAP provides insight into:

* Which features are most important.
* How strongly each feature affects predictions.
* Whether features push predictions higher or lower.
* How individual borough predictions are influenced.

### Feature Importance

The SHAP analysis identified the following approximate ranking:

1. **Employment score** — ~1.90
2. **Income score** — ~1.55
3. **Crime domain score** — ~1.05
4. **Health score** — ~0.45
5. Education score
6. Total crime count
7. Environment score

Employment deprivation was therefore the strongest predictor in the model.

Importantly, this does **not** prove that employment deprivation causes overall deprivation. The model identifies predictive relationships rather than causal relationships.

---

## Key Findings

### Employment is a major predictor

Employment-related deprivation had the strongest influence on the model's predictions.

This suggests that access to employment and labour-market conditions are closely associated with broader deprivation patterns across London's boroughs.

### Income is strongly connected to deprivation

Income deprivation was also one of the strongest predictors and showed an extremely strong correlation with overall deprivation.

### Crime count is not the same as deprivation

Raw crime counts were less strongly associated with deprivation than expected.

Westminster demonstrates why raw crime totals can be misleading: areas with large numbers of visitors and workers can experience high recorded crime without having similarly high levels of resident deprivation.

### Clear geographical pattern

The analysis identified a broad **east/inner-east versus south-west/outer London pattern**.

Barking and Dagenham and Hackney appeared among the most deprived boroughs, while Richmond upon Thames, Kingston upon Thames and Sutton appeared among the least deprived in the analysis.

### Borough averages hide local inequality

A borough-level average cannot capture every neighbourhood's circumstances.

Kensington and Chelsea is a useful example because significant differences can exist between wealthy and deprived areas within the same borough.

### Education was less influential than expected

Education did not appear among the strongest SHAP features.

This does **not** mean education is unimportant. It indicates that, within this dataset and model, education contributed less to the model's predictions than variables such as employment and income.

---

## Visualisations

The project includes analysis and visualisations such as:

* Correlation heatmap
* Feature importance chart
* SHAP summary/dot plot
* Borough deprivation ranking
* Employment vs deprivation scatter plot
* Income vs health scatter plot
* Interactive London borough map
* Geospatial deprivation visualisation

---

## Technology Stack

### Programming

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest Regressor

### Explainable AI

* SHAP

### Visualisation

* Matplotlib
* Seaborn
* Plotly

### Geospatial Analysis

* GeoPandas
* London borough boundary data

### Dashboard

* Interactive geospatial dashboard

---

## Project Structure

```text
london-deprivation-analysis/
│
├── data/
│   ├── deprivation/
│   ├── crime/
│   └── boundaries/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── exploratory_analysis.ipynb
│   └── machine_learning.ipynb
│
├── dashboard/
│   └── app.py
│
├── visualisations/
│
├── requirements.txt
│
└── README.md
```

---

## Limitations

There are several important limitations to this analysis.

### Small sample size

The final dataset contains only **32 boroughs**. This limits the statistical power of the model and increases the risk of overfitting.

### Borough-level aggregation

Aggregating LSOA data into borough averages can hide substantial inequalities between neighbourhoods.

### Correlation is not causation

The model identifies predictive relationships. It cannot establish that a particular factor directly causes deprivation.

### Crime measurement

Total recorded crime is affected by population, tourism, commuting and visitor footfall. Raw crime counts therefore do not necessarily represent the experience of deprivation among residents.

### Data coverage

The project combines datasets from different sources and time periods. Differences in measurement and timing may affect relationships between variables.

---

## Overall Conclusion

The analysis demonstrates that London's deprivation is not explained by a single factor.

Economic, employment, health, education, crime and environmental conditions interact to produce different deprivation patterns across the city.

The machine learning model achieved strong predictive performance on the available dataset, while SHAP made it possible to examine which variables were driving those predictions.

The most important finding was the strong role of **employment and income-related deprivation**.

At the same time, the geographical analysis demonstrated that deprivation is unevenly distributed across London and that borough-level averages can conceal substantial differences between neighbourhoods.

The project therefore demonstrates how **machine learning + explainable AI + geospatial visualisation** can be combined to make socioeconomic analysis more accessible and interpretable.

---

## Research Context

This project was developed as an AI-driven analytical framework for understanding deprivation across London's boroughs. It builds on existing deprivation research while focusing specifically on the combination of:

* Official multi-domain deprivation data
* Metropolitan Police crime data
* Random Forest machine learning
* SHAP explainability
* Borough-level geospatial analysis

The approach aims to make socioeconomic analysis accessible not only to researchers and policymakers, but also to wider audiences interested in understanding inequality across London.
