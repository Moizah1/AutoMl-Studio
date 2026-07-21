import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------------------------------
# App config & constants
# --------------------------------------------------------------------------
st.set_page_config(page_title="Data God - AutoML Studio", page_icon="🤖", layout="wide")

SOURCE_FILE = "sourcedata.csv"
MODEL_FILE = "best_model.pkl"

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_source_data():
    if os.path.exists(SOURCE_FILE):
        try:
            return pd.read_csv(SOURCE_FILE, index_col=False)
        except Exception:
            return None
    return None


def save_source_data(dataframe: pd.DataFrame):
    dataframe.to_csv(SOURCE_FILE, index=False)


def missing_value_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing = dataframe.isna().sum()
    pct = (missing / len(dataframe) * 100).round(2)
    summary = pd.DataFrame({"dtype": dataframe.dtypes.astype(str), "missing_count": missing, "missing_%": pct})
    return summary.sort_values("missing_count", ascending=False)


def build_preprocessor(X: pd.DataFrame, normalize: bool):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if normalize:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers)


def get_feature_importance(fitted_pipeline: Pipeline):
    """Best-effort extraction of feature importances / coefficients for a bar chart."""
    try:
        model = fitted_pipeline.named_steps["model"]
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.ravel(model.coef_)
        else:
            return None

        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df["abs_importance"] = imp_df["importance"].abs()
        return imp_df.sort_values("abs_importance", ascending=False).head(20)
    except Exception:
        return None


# Session state initialization
if "df" not in st.session_state:
    st.session_state.df = load_source_data()
if "trained_task" not in st.session_state:
    st.session_state.trained_task = None  # "Regression" or "Classification"
if "trained_target" not in st.session_state:
    st.session_state.trained_target = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None

df = st.session_state.df

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 I'm the Data God....!")

    logo_path = "databot-icon.png"
    if os.path.exists(logo_path):
        try:
            st.image(Image.open(logo_path), use_container_width=True)
        except Exception:
            pass
    else:
        st.caption("💡 Tip: place `databot-icon.png` next to app.py to show a logo here.")

    choice = st.radio(
        "Navigation",
        ["Home", "Upload", "Data Cleaning", "Profile Report", "Model Training", "Predict"],
    )

    st.divider()
    if df is not None:
        st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        if st.button("🗑️ Clear Dataset"):
            st.session_state.df = None
            st.session_state.trained_task = None
            st.session_state.trained_target = None
            st.session_state.feature_columns = None
            if os.path.exists(SOURCE_FILE):
                os.remove(SOURCE_FILE)
            st.rerun()
    else:
        st.info("No dataset loaded yet.")

# --------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------
if choice == "Home":
    st.title("Let me Do your Work 🤖")
    st.markdown(
        """
Welcome! This app walks you through a full mini machine-learning pipeline:

1. **Upload** — bring in a CSV dataset.
2. **Data Cleaning** — handle missing values, drop columns, remove duplicates.
3. **Profile Report** — explore distributions, correlations, and missing data.
4. **Model Training** — train and compare regression/classification models.
5. **Predict** — use your trained model on new data.

Use the navigation panel on the left to get started.

*Built on scikit-learn, pandas, and plotly — fully compatible with Python 3.13.*
        """
    )

# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------
if choice == "Upload":
    st.title("Push Your Dataset Here ✌️")
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        try:
            uploaded_df = pd.read_csv(file, index_col=False)
            st.session_state.df = uploaded_df
            save_source_data(uploaded_df)
            st.session_state.trained_task = None
            st.session_state.trained_target = None
            st.session_state.feature_columns = None
            df = uploaded_df
            st.success("Dataset uploaded successfully!")
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(50), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing cells", int(df.isna().sum().sum()))

        st.subheader("Column Overview")
        st.dataframe(missing_value_summary(df), use_container_width=True)

        st.download_button(
            "Download current dataset as CSV",
            data=df.to_csv(index=False),
            file_name="dataset.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------
# Data Cleaning
# --------------------------------------------------------------------------
if choice == "Data Cleaning":
    st.title("Clean Your Data 🧹")
    if df is None:
        st.warning("Please upload the dataset first ⚠️")
    else:
        working_df = df.copy()

        st.subheader("1. Drop columns")
        cols_to_drop = st.multiselect("Select columns to remove", working_df.columns)
        if cols_to_drop:
            working_df = working_df.drop(columns=cols_to_drop)

        st.subheader("2. Handle missing values")
        strategy = st.selectbox(
            "Strategy",
            ["Do nothing", "Drop rows with any missing values", "Fill numeric with mean",
             "Fill numeric with median", "Fill with mode (all columns)", "Fill with a constant"],
        )
        if strategy == "Drop rows with any missing values":
            working_df = working_df.dropna()
        elif strategy == "Fill numeric with mean":
            num_cols = working_df.select_dtypes(include=np.number).columns
            working_df[num_cols] = working_df[num_cols].fillna(working_df[num_cols].mean())
        elif strategy == "Fill numeric with median":
            num_cols = working_df.select_dtypes(include=np.number).columns
            working_df[num_cols] = working_df[num_cols].fillna(working_df[num_cols].median())
        elif strategy == "Fill with mode (all columns)":
            for c in working_df.columns:
                mode_vals = working_df[c].mode()
                if not mode_vals.empty:
                    working_df[c] = working_df[c].fillna(mode_vals.iloc[0])
        elif strategy == "Fill with a constant":
            const_val = st.text_input("Constant value to fill with", "0")
            working_df = working_df.fillna(const_val)

        st.subheader("3. Remove duplicates")
        if st.checkbox("Drop duplicate rows"):
            working_df = working_df.drop_duplicates()

        st.subheader("Preview after cleaning")
        st.dataframe(working_df.head(50), use_container_width=True)
        st.caption(f"Shape: {working_df.shape[0]} rows × {working_df.shape[1]} columns")

        if st.button("✅ Apply changes to working dataset"):
            st.session_state.df = working_df
            save_source_data(working_df)
            st.success("Cleaning applied! This is now your active dataset.")
            st.rerun()

# --------------------------------------------------------------------------
# Profile Report (lightweight, custom - no ydata-profiling dependency)
# --------------------------------------------------------------------------
if choice == "Profile Report":
    if df is not None:
        st.title("Exploratory Data Analysis")

        st.subheader("Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Duplicate rows", int(df.duplicated().sum()))
        c4.metric("Missing cells", int(df.isna().sum().sum()))

        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        st.subheader("Missing values")
        miss = missing_value_summary(df)
        if miss["missing_count"].sum() > 0:
            fig = px.bar(miss.reset_index(), x="index", y="missing_count", labels={"index": "column"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No missing values found 🎉")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        if numeric_cols:
            st.subheader("Numeric distributions")
            selected_num = st.selectbox("Choose a numeric column", numeric_cols)
            fig = px.histogram(df, x=selected_num, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

            if len(numeric_cols) > 1:
                st.subheader("Correlation heatmap")
                corr = df[numeric_cols].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)

        if categorical_cols:
            st.subheader("Categorical breakdown")
            selected_cat = st.selectbox("Choose a categorical column", categorical_cols)
            counts = df[selected_cat].value_counts().head(20).reset_index()
            counts.columns = [selected_cat, "count"]
            fig = px.bar(counts, x=selected_cat, y="count")
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download summary statistics as CSV",
            data=df.describe(include="all").transpose().to_csv(),
            file_name="summary_statistics.csv",
            mime="text/csv",
        )
    else:
        st.warning("Please upload the dataset first ⚠️")

# --------------------------------------------------------------------------
# Model Training (scikit-learn based)
# --------------------------------------------------------------------------
if choice == "Model Training":
    if df is not None:
        st.title("Train a Model 🚀")

        target = st.selectbox("Select the Target Feature in the Dataset", df.columns)
        task_type = st.selectbox("Select the Problem Type", ("Regression", "Classification"))

        with st.expander("Advanced options"):
            ignore_cols = st.multiselect(
                "Columns to ignore during training", [c for c in df.columns if c != target]
            )
            train_size = st.slider("Train set size", 0.5, 0.95, 0.7, 0.05)
            normalize = st.checkbox("Normalize numeric features", value=True)
            random_state = st.number_input("Random seed (for reproducibility)", value=42, step=1)
            clear_after_training = st.checkbox("Delete source CSV after training", value=False)

        if st.button("Train the Model"):
            try:
                feature_cols = [c for c in df.columns if c != target and c not in ignore_cols]
                clean_df = df[feature_cols + [target]].dropna(subset=[target])
                X = clean_df[feature_cols]
                y = clean_df[target]

                if task_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
                    st.error("The selected target isn't numeric. Choose Classification instead, or pick a numeric target.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=train_size, random_state=int(random_state),
                    stratify=y if task_type == "Classification" and y.nunique() > 1 else None,
                )

                candidates = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS
                results = []
                fitted_pipelines = {}

                progress = st.progress(0, text="Training models...")
                for i, (name, model) in enumerate(candidates.items()):
                    preprocessor = build_preprocessor(X_train, normalize)
                    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)

                    if task_type == "Regression":
                        row = {
                            "Model": name,
                            "R2": r2_score(y_test, preds),
                            "MAE": mean_absolute_error(y_test, preds),
                            "RMSE": mean_squared_error(y_test, preds) ** 0.5,
                        }
                        primary_metric = "R2"
                    else:
                        row = {
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, preds),
                            "F1 (macro)": f1_score(y_test, preds, average="macro", zero_division=0),
                            "Precision (macro)": precision_score(y_test, preds, average="macro", zero_division=0),
                            "Recall (macro)": recall_score(y_test, preds, average="macro", zero_division=0),
                        }
                        primary_metric = "F1 (macro)"

                    results.append(row)
                    fitted_pipelines[name] = pipeline
                    progress.progress((i + 1) / len(candidates), text=f"Trained {name}")

                progress.empty()

                compare_df = pd.DataFrame(results).sort_values(primary_metric, ascending=False).reset_index(drop=True)
                st.subheader("Model Comparison")
                st.dataframe(compare_df, use_container_width=True)

                best_name = compare_df.iloc[0]["Model"]
                best_pipeline = fitted_pipelines[best_name]
                st.success(f"🏆 Best model: **{best_name}**")

                imp_df = get_feature_importance(best_pipeline)
                if imp_df is not None:
                    st.subheader("Feature importance (top 20)")
                    fig = px.bar(imp_df, x="importance", y="feature", orientation="h")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("(Feature importance not available for this model type.)")

                joblib.dump(best_pipeline, MODEL_FILE)
                st.session_state.trained_task = task_type
                st.session_state.trained_target = target
                st.session_state.feature_columns = feature_cols

                with open(MODEL_FILE, "rb") as f:
                    st.download_button("Download Model", f, file_name="best_model.pkl")

                if clear_after_training and os.path.exists(SOURCE_FILE):
                    os.remove(SOURCE_FILE)
                    st.info("Source CSV removed as requested. Your session data is still available.")

                st.success("Training complete! Head to the Predict page to try it out.")

            except Exception as e:
                st.error(f"Training failed: {e}")
    else:
        st.warning("Please upload the dataset first ⚠️")

# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
if choice == "Predict":
    st.title("Run Predictions 🔮")

    if not os.path.exists(MODEL_FILE) or st.session_state.trained_task is None:
        st.warning("Please train a model first on the Model Training page ⚠️")
    else:
        st.info(
            f"Using your trained **{st.session_state.trained_task}** model "
            f"(target: `{st.session_state.trained_target}`)."
        )
        pred_file = st.file_uploader("Upload new data for prediction (CSV)", type=["csv"], key="predict_upload")

        if pred_file is not None:
            try:
                new_df = pd.read_csv(pred_file, index_col=False)
                feature_cols = st.session_state.feature_columns
                missing_cols = [c for c in feature_cols if c not in new_df.columns]
                if missing_cols:
                    st.error(f"Uploaded data is missing required columns: {missing_cols}")
                else:
                    loaded_pipeline = joblib.load(MODEL_FILE)
                    with st.spinner("Predicting..."):
                        predictions = loaded_pipeline.predict(new_df[feature_cols])

                    result_df = new_df.copy()
                    result_df[f"predicted_{st.session_state.trained_target}"] = predictions

                    st.subheader("Predictions")
                    st.dataframe(result_df, use_container_width=True)

                    st.download_button(
                        "Download Predictions",
                        data=result_df.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
