import streamlit as st
import pandas as pd
import pydeck as pdk
from PIL import Image
from pathlib import Path

# =========================
# Paths (robust)
# =========================
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = ROOT / "data"
OUTPUT_IMG_DIR = ROOT / "output" / "images"

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Melbourne Market Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Melbourne Housing Market Analytics")

st.info("""
💡 **Key Insight:** Through Random Forest analysis, we discovered that **Latitude (North-South Divide)** is a stronger driver of house prices than raw **Distance to CBD**. 
Properties in the South/East (Toorak, South Yarra) command a premium that defies simple distance logic.
""")

# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    """
    Try common locations for melb_data.csv and load it robustly.
    Also normalises column names and ensures numeric types.
    """
    candidates = [
        ROOT / "melb_data.csv",
        DATA_DIR / "melb_data.csv",
        DATA_DIR / "processed" / "melb_data.csv",
        DATA_DIR / "raw" / "melb_data.csv",
    ]

    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        st.error(
            "Cannot find melb_data.csv. Tried:\n- " +
            "\n- ".join(str(p) for p in candidates)
        )
        st.stop()

    df = pd.read_csv(csv_path)

    # ---- Column name compatibility (Latitude/Longitude common typos) ----
    # Your dataset uses "Lattitude" / "Longtitude" (typos) in many Melbourne housing datasets.
    col_map = {}
    if "Lattitude" in df.columns and "Latitude" not in df.columns:
        col_map["Lattitude"] = "Latitude"
    if "Longtitude" in df.columns and "Longitude" not in df.columns:
        col_map["Longtitude"] = "Longitude"
    if col_map:
        df = df.rename(columns=col_map)

    # Required columns
    required = ["Latitude", "Longitude", "Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")
        st.stop()

    # Force numeric for coordinates/price (robust against strings)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # ---- Unit_Price creation (optional but recommended) ----
    # If not present, compute from land size. Clean it thoroughly to avoid NAType issues downstream.
    if "Unit_Price" not in df.columns:
        land_candidates = ["Landsize", "LandSize", "Land_Size", "landsize"]
        land_col = next((c for c in land_candidates if c in df.columns), None)

        if land_col is not None:
            df[land_col] = pd.to_numeric(df[land_col], errors="coerce")
            df["Unit_Price"] = df["Price"] / df[land_col]
            # Remove invalid land sizes (<=0) -> Unit_Price becomes NA
            df.loc[df[land_col] <= 0, "Unit_Price"] = pd.NA
        else:
            # Fallback if no land size: use Price (still allows visual encoding)
            df["Unit_Price"] = df["Price"]

    # Ensure Unit_Price numeric
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

    # Basic cleaning
    df = df.dropna(subset=["Latitude", "Longitude", "Price"])
    df = df[df["Price"] < 3_000_000]

    # Optional: keep only plausible Melbourne bounding box (extra safety)
    # If this filters too much, comment it out.
    df = df[df["Latitude"].between(-38.5, -37.3) & df["Longitude"].between(144.3, 145.6)]

    return df


df = load_data()

# =========================
# Debug (collapsed)
# =========================
with st.expander("Debug (columns & coordinate stats)", expanded=False):
    st.write("Columns:", list(df.columns))
    st.write(df[["Latitude", "Longitude"]].describe())

# =========================
# Sidebar filters
# =========================
st.sidebar.header("🔍 Filter Data")
price_filter = st.sidebar.slider(
    "Max Price (Budget)",
    int(df["Price"].min()),
    int(df["Price"].max()),
    1_500_000
)

# Filter + optional sampling (for performance)
filtered_df = df[df["Price"] <= price_filter].copy()

# Optional: sample to keep map smooth
if len(filtered_df) > 3000:
    filtered_df = filtered_df.sample(3000, random_state=42)

# =========================
# Layout
# =========================
col1, col2 = st.columns([2, 1])

# =========================
# Left: PyDeck colour map
# =========================
with col1:
    st.subheader(f"📍 Price Heatmap ({len(filtered_df)} Properties)")

    # Ensure required fields exist & are numeric
    filtered_df["Unit_Price"] = pd.to_numeric(filtered_df["Unit_Price"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["Latitude", "Longitude", "Unit_Price"]).copy()

    if len(filtered_df) == 0:
        st.warning("No valid records after cleaning Unit_Price/coordinates.")
        st.stop()

    # Winsorise + normalise (avoid outliers breaking colour scale)
    low = filtered_df["Unit_Price"].quantile(0.05)
    high = filtered_df["Unit_Price"].quantile(0.95)
    v = filtered_df["Unit_Price"].clip(lower=low, upper=high)

    denom = (v.max() - v.min())
    if denom == 0:
        filtered_df["_v"] = 0.5
    else:
        filtered_df["_v"] = (v - v.min()) / (denom)

    # Final NA protection
    filtered_df["_v"] = filtered_df["_v"].fillna(0.5)

    # Colour: low -> blue, high -> red
    filtered_df["_r"] = (255 * filtered_df["_v"]).round().astype("int32")
    filtered_df["_g"] = (80 * (1 - filtered_df["_v"])).round().astype("int32")
    filtered_df["_b"] = (255 * (1 - filtered_df["_v"])).round().astype("int32")

    # PyDeck layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_df,
        get_position="[Longitude, Latitude]",
        get_fill_color="[_r, _g, _b, 140]",
        get_radius="200 + 800 * _v",
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.ViewState(
        latitude=float(filtered_df["Latitude"].mean()),
        longitude=float(filtered_df["Longitude"].mean()),
        zoom=10,
        pitch=0
    )

    tooltip = {
        "html": "<b>Suburb:</b> {Suburb}<br/>"
                "<b>Price:</b> ${Price}<br/>"
                "<b>Unit Price:</b> {Unit_Price}",
        "style": {"backgroundColor": "white", "color": "black"}
    }

    # NOTE: no map_style to avoid Mapbox token dependency
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip
        ),
        use_container_width=True
    )

    st.caption("Warmer/larger points indicate higher Land Value Density (Price/sqm).")

# =========================
# Right: Feature importance image + commentary
# =========================
with col2:
    st.subheader("🏆 What drives the price?")

    img_candidates = [
        ROOT / "feature_importance.png",
        OUTPUT_IMG_DIR / "feature_importance.png",
    ]
    img_path = next((p for p in img_candidates if p.exists()), None)

    if img_path:
        image = Image.open(img_path)
        st.image(image, caption="Feature Importance (Random Forest)", use_container_width=True)
    else:
        st.warning(
            "Feature Importance image not found. Tried:\n- " +
            "\n- ".join(str(p) for p in img_candidates)
        )

    st.markdown("---")
    st.markdown("""
**Analysis:**
* **Latitude (#1):** The specific neighborhood "prestige" (South vs North).
* **Landsize:** Intrinsic asset value.
* **Distance:** Still important, but secondary to the specific "Zone".
""")

# =========================
# Bottom KPIs
# =========================
st.markdown("---")
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric("Average Price (Filtered)", f"${filtered_df['Price'].mean():,.0f}")

if "Suburb" in filtered_df.columns and len(filtered_df) > 0:
    kpi2.metric("Most Expensive Suburb", filtered_df.loc[filtered_df["Price"].idxmax()]["Suburb"])
else:
    kpi2.metric("Most Expensive Suburb", "N/A")

kpi3.metric("Data Sample Size", f"{len(filtered_df)} Records")
