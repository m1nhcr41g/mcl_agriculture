import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Đánh giá mô hình dự đoán tưới tiêu với mô hình LightGBM với bộ dữ liệu của nông nghiệp Ấn Độ",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/model_lightgbm.pkl")

clf = load_model()

def create_advanced_features(data):
    df_fe = data.copy()
    df_fe["ET_Index"] = (df_fe["Temperature_C"] * df_fe["Sunlight_Hours"] * (1 + 0.1 * df_fe["Wind_Speed_kmh"])) / (df_fe["Humidity"] + 1)
    df_fe["Moisture_Deficit"] = (df_fe["Soil_Moisture"] / (df_fe["Rainfall_mm"] + 1))
    df_fe["Heat_Humidity_Ratio"] = (df_fe["Temperature_C"] / (df_fe["Humidity"] + 1))
    df_fe["Thermal_Radiation"] = (df_fe["Temperature_C"] * df_fe["Sunlight_Hours"])
    return df_fe

st.title("Thử nghiệm và đánh giá mô hình dự đoán tưới tiêu với mô hình LightGBM với bộ dữ liệu của nông nghiệp Ấn Độ")


with st.expander("Hướng dẫn và bảng chú thích thuật ngữ Tiếng Việt"):
    st.markdown("""
    | Thuật ngữ | Dịch nghĩa Tiếng Việt | Ghi chú giải thích |
    | :--- | :--- | :--- |
    | **Crop Type** | Loại cây trồng | Cotton (Bông), Maize (Ngô), Potato (Khoai tây), Rice (Lúa), Sugarcane (Mía), Wheat (Lúa mì) |
    | **Crop Growth Stage** | Giai đoạn sinh trưởng | Sowing (Gieo hạt), Vegetative (Phát triển thân lá), Flowering (Ra hoa), Harvest (Thu hoạch) |
    | **Season** | Mùa vụ | Kharif (Mùa hè/Mưa), Rabi (Mùa đông/Khô), Zaid (Mùa hè phụ/Ngắn ngày) |
    | **Irrigation Type** | Loại hình tưới | Canal (Tưới kênh), Drip (Tưới nhỏ giọt), Rainfed (Nhờ nước mưa), Sprinkler (Tưới phun mưa) |
    | **Water Source** | Nguồn nước | Groundwater (Nước ngầm), Rainwater (Nước mưa), Reservoir (Hồ chứa), River (Nước sông) |
    | **Soil Type** | Loại đất | Clay (Đất sét), Loamy (Đất thịt), Sandy (Đất cát), Silt (Đất phù sa) |
    | **Mulching Used** | Sử dụng lớp phủ | Yes (Có phủ bạt/rơm rạ), No (Không phủ) |
    | **Region** | Khu vực địa lý | Central (Miền Trung), East (Miền Đông), North (Miền Bắc), South (Miền Nam), West (Miền Tây) |
    """)

tab1, tab2, tab3 = st.tabs(["1. Thời Tiết và Môi Trường", "2. Đất Đai và Cây Trồng", "3. Tưới Tiêu và Khu Vực"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.slider("Nhiệt độ (°C)", 10.0, 50.0, 25.0, 0.5)
        humidity = st.slider("Độ ẩm không khí (%)", 10.0, 100.0, 75.0, 1.0)
    with col2:
        rainfall = st.slider("Lượng mưa (mm)", 0.0, 2500.0, 500.0, 0.5)
        sunlight = st.slider("Số giờ nắng (giờ)", 0.0, 14.0, 5.0, 0.5)
    with col3:
        wind_speed = st.slider("Tốc độ gió (km/h)", 0.0, 25.0, 8.0, 0.5)
        soil_ph = st.slider("Độ pH đất", 3.0, 10.0, 6.8, 0.1)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        soil_moisture = st.slider("Độ ẩm đất (%)", 0.0, 100.0, 42.0, 0.5)
        organic_carbon = st.slider("Carbon hữu cơ (%)", 0.0, 3.0, 1.0, 0.01)
    with col2:
        electrical_conductivity = st.slider("Độ dẫn điện đất", 0.0, 4.0, 1.5, 0.01)
    with col3:
        field_area = st.slider("Diện tích ruộng (ha)", 0.0, 20.0, 5.0, 0.01)

with tab3:
    crop_map = {"Cotton": 0, "Maize": 1, "Potato": 2, "Rice": 3, "Sugarcane": 4, "Wheat": 5}
    growth_stage_map = {"Flowering": 0, "Harvest": 1, "Sowing": 2, "Vegetative": 3}
    season_map = {"Kharif": 0, "Rabi": 1, "Zaid": 2}
    irrigation_map = {"Canal": 0, "Drip": 1, "Rainfed": 2, "Sprinkler": 3}
    water_source_map = {"Groundwater": 0, "Rainwater": 1, "Reservoir": 2, "River": 3}
    soil_map = {"Clay": 0, "Loamy": 1, "Sandy": 2, "Silt": 3}
    mulching_map = {"No": 0, "Yes": 1}
    region_map = {"Central": 0, "East": 1, "North": 2, "South": 3, "West": 4}

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_soil = st.selectbox("Loại đất", options=list(soil_map))
        selected_crop = st.selectbox("Loại cây trồng", options=list(crop_map))
    with col2:
        selected_stage = st.selectbox("Giai đoạn sinh trưởng", options=list(growth_stage_map))
        selected_season = st.selectbox("Mùa vụ", options=list(season_map))
    with col3:
        selected_irrigation = st.selectbox("Loại hình tưới", options=list(irrigation_map))
        selected_water_source = st.selectbox("Nguồn nước", options=list(water_source_map))

    col1, col2, col3 = st.columns(3)
    with col1:
        previous_irrigation = st.slider("Lượng tưới trước đó (mm)", 0.0, 150.0, 30.0, 0.1)
    with col2:
        mulching_used = st.selectbox("Sử dụng lớp phủ", options=list(mulching_map))
    with col3:
        selected_region = st.selectbox("Khu vực", options=list(region_map))

st.divider()

if st.button("Chạy Dự Đoán Mô Hình LightGBM", use_container_width=True, type="primary"):
    input_dict = {
        "Soil_Type": soil_map[selected_soil],
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": electrical_conductivity,
        "Temperature_C": temp,
        "Humidity": humidity,
        "Sunlight_Hours": sunlight,
        "Wind_Speed_kmh": wind_speed,
        "Rainfall_mm": rainfall,
        "Crop_Type": crop_map[selected_crop],
        "Crop_Growth_Stage": growth_stage_map[selected_stage],
        "Season": season_map[selected_season],
        "Irrigation_Type": irrigation_map[selected_irrigation],
        "Water_Source": water_source_map[selected_water_source],
        "Field_Area_hectare": field_area,
        "Mulching_Used": mulching_map[mulching_used],
        "Previous_Irrigation_mm": previous_irrigation,
        "Region": region_map[selected_region]
    }

    raw_df = pd.DataFrame([input_dict])
    processed_df = create_advanced_features(raw_df)
    model_features = [
        "Soil_Type", "Soil_pH", "Soil_Moisture", "Organic_Carbon",
        "Electrical_Conductivity", "Temperature_C", "Humidity", "Rainfall_mm",
        "Sunlight_Hours", "Wind_Speed_kmh", "Crop_Type", "Crop_Growth_Stage",
        "Season", "Irrigation_Type", "Water_Source", "Field_Area_hectare",
        "Mulching_Used", "Previous_Irrigation_mm", "Region", "ET_Index",
        "Moisture_Deficit", "Heat_Humidity_Ratio", "Thermal_Radiation"
    ]
    processed_df = processed_df.reindex(columns=model_features)

    label_map = {0: "LOW (Thấp)", 1: "MEDIUM (Trung Bình)", 2: "HIGH (Cao)"}
    color_map = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"}

    pred_class = int(clf.predict(processed_df)[0])

    st.subheader("Kết Quả Dự Đoán")
    st.markdown(
        f"""
        <div style="background-color: {color_map[pred_class]}; padding: 35px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0; font-weight: 500;">Mức Nhu Cầu Tưới Dự Đoán</h2>
            <h1 style="color: white; margin: 15px 0 0 0; font-size: 42px; font-weight: bold;">{label_map[pred_class]}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )