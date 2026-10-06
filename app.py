import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_REPO_ID = "masadullahna/melb-by-na"
HF_FILENAME = "melb-price-model.joblib"

TYPE_LABELS = {"h": "House", "t": "Townhouse", "u": "Unit"}
TYPE_ORDER = ["h", "t", "u"]

METHOD_LABELS = {
    "S": "Sold",
    "SP": "Sold prior",
    "PI": "Passed in",
    "VB": "Vendor bid",
    "SA": "Sold after auction",
}
METHOD_ORDER = ["S", "SP", "PI", "VB", "SA"]


st.set_page_config(
    page_title="Melbourne house price predictor",
    page_icon=":material/home:",
    layout="centered",
)

st.title("Melbourne house price predictor", icon=":material/home:")


@st.cache_resource
def load_bundle():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return joblib.load(model_path)


try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load the model from `huggingface.co/{HF_REPO_ID}`: {exc}")
    st.stop()

if not isinstance(bundle, dict) or "models" not in bundle:
    st.error(
        f"`{HF_REPO_ID}` bundle is still in the old single-model format. "
        "Re-run `notebooks/exploration.ipynb` to rebuild it with every algorithm."
    )
    st.stop()

models = bundle["models"]
metrics = bundle["metrics"]

reference_model = next(iter(models.values()))
category_values = dict(
    zip(
        reference_model.named_steps["preprocessor"].transformers_[1][2],
        reference_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .categories_,
    )
)
suburbs = sorted(category_values["Suburb"])
regions = sorted(category_values["Regionname"])


model_names = list(models)
default_index = (
    model_names.index("Random Forest") if "Random Forest" in model_names else 0
)

selected_name = st.selectbox(
    "Algorithm",
    model_names,
    index=default_index,
    help="Choose which trained model makes the prediction. "
    "The metrics below update to match.",
)
model = models[selected_name]
selected_metrics = metrics[selected_name]

st.caption(
    f"**{selected_name}** · holdout R² {selected_metrics['R2']:.3f} · "
    f"MAE ≈ ${selected_metrics['MAE']:,.0f} · "
    f"RMSE ≈ ${selected_metrics['RMSE']:,.0f} · "
    "trained on 13,580 Melbourne sales"
)

with st.expander("Compare all models on the holdout test set"):
    report = pd.DataFrame(metrics).T.sort_values("R2", ascending=False)
    comparison = pd.DataFrame(
        {
            "Selected": ["✓" if name == selected_name else "" for name in report.index],
            "R²": report["R2"].map("{:.3f}".format),
            "MAE": report["MAE"].map("${:,.0f}".format),
            "RMSE": report["RMSE"].map("${:,.0f}".format),
        },
        index=report.index,
    )
    comparison.index.name = "Algorithm"
    st.dataframe(comparison, width="stretch")

with st.form("prediction_form"):
    location_col, property_col = st.columns(2)

    with location_col:
        st.subheader("Location")

        suburb = st.selectbox(
            "Suburb",
            suburbs,
            index=None,
            placeholder="Select a suburb",
        )

        region = st.selectbox(
            "Region",
            regions,
            index=None,
            placeholder="Select a region",
        )

        postcode = st.number_input(
            "Postcode",
            min_value=3000,
            max_value=3977,
            value=3000,
            step=1,
        )

        distance = st.number_input(
            "Distance from CBD (km)",
            min_value=0.0,
            max_value=48.1,
            value=10.0,
            step=0.1,
            format="%.1f",
        )

        latitude = st.number_input(
            "Latitude",
            min_value=-38.18255,
            max_value=-37.40853,
            value=-37.81,
            format="%.5f",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=144.43181,
            max_value=145.52635,
            value=144.96,
            format="%.5f",
        )

        propertycount = st.number_input(
            "Properties in suburb",
            min_value=249,
            max_value=21650,
            value=1000,
            step=1,
        )

    with property_col:
        st.subheader("Property")

        prop_type = st.segmented_control(
            "Property type",
            TYPE_ORDER,
            format_func=lambda code: TYPE_LABELS[code],
            default="h",
            required=True,
        )

        method = st.segmented_control(
            "Sale method",
            METHOD_ORDER,
            format_func=lambda code: METHOD_LABELS[code],
            default="S",
            required=True,
        )

        rooms = st.number_input(
            "Rooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        bedroom2 = st.number_input(
            "Bedrooms (Bedroom2)",
            min_value=0,
            max_value=20,
            value=3,
            step=1,
        )

        bathroom = st.number_input(
            "Bathrooms",
            min_value=0,
            max_value=8,
            value=1,
            step=1,
        )

        car = st.number_input(
            "Car spaces",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

        landsize = st.number_input(
            "Land size (m²)",
            min_value=0.0,
            max_value=433014.0,
            value=500.0,
            step=10.0,
            format="%.0f",
        )

    submitted = st.form_submit_button(
        "Predict house price",
        type="primary",
        icon=":material/payments:",
        width="stretch",
    )


if submitted:
    if suburb is None or region is None:
        st.info("Select a suburb and a region to get a prediction.")
    else:
        input_data = pd.DataFrame(
            [
                {
                    "Suburb": suburb,
                    "Rooms": rooms,
                    "Type": prop_type,
                    "Method": method,
                    "Distance": distance,
                    "Postcode": postcode,
                    "Bedroom2": bedroom2,
                    "Bathroom": bathroom,
                    "Car": car,
                    "Landsize": landsize,
                    "Lattitude": latitude,
                    "Longtitude": longitude,
                    "Regionname": region,
                    "Propertycount": propertycount,
                }
            ],
            columns=model.feature_names_in_,
        )

        try:
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            st.success("Prediction complete")
            st.metric(
                label="Estimated house price",
                value=f"${prediction:,.0f}",
            )
