# Hệ thống dự đoán lượng nước cần tưới cho nông nghiệp
Ứng dụng Web hỗ trợ ra quyết định tưới tiêu tự động dựa trên mô hình học máy lightgbm, phân tích 19 thông số chi tiết về môi trường, đất đai, cây trồng và khu vực nông nghiệp. Sau khi đã chạy với 3 mô hình là lightgbm, random forest, lstm thì nhận thấy chỉ số là tốt nhất nên đã quyết định chọn lightgbm
## 1. Giới Thiệu Đề Tài
Đề tài tập trung xây dựng một hệ thống học máy có khả năng:
Phân tích các yếu tố khí hậu, chất lượng đất và đặc thù sinh trưởng của cây trồng.
Dự đoán chính xác nhu cầu tưới tiêu theo 3 mức độ: **THẤP (LOW)**, **TRUNG BÌNH (MEDIUM)**, và **CAO (HIGH)**.
Sử dụng mô hình lightgbm (Light Gradient Boosting Machine) làm thuật toán cốt lõi.
## 2. Kiến trúc và đặc trưng Dữ Liệu (19 Features)

Mô hình tiếp nhận 19 thông số đầu vào thuộc bộ dữ liệu từ: https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset/data:

**Thời tiết & Khí hậu:** Nhiệt độ (`Temperature_C`), Độ ẩm không khí (`Humidity`), Lượng mưa (`Rainfall_mm`), Số giờ nắng (`Sunlight_Hours`), Tốc độ gió (`Wind_Speed_kmh`).
**Thổ thổ & Đất đai:** Độ ẩm đất (`Soil_Moisture`), Độ pH đất (`Soil_pH`), Carbon hữu cơ (`Organic_Carbon`), Độ dẫn điện đất (`Electrical_Conductivity`), Loại đất (`Soil_Type`: *Clay, Loamy, Sandy, Silt*).
**Cây trồng & Canh tác:** Loại cây (`Crop_Type`: *Cotton, Maize, Potato, Rice, Sugarcane, Wheat*), Giai đoạn sinh trưởng (`Crop_Growth_Stage`: *Flowering, Harvest, Sowing, Vegetative*), Lượng tưới trước đó (`Previous_Irrigation_mm`), Lớp phủ đất (`Mulching_Used`: *Yes/No*), Diện tích ruộng (`Field_Area_hectare`).
**Mùa vụ & Địa lý:** Mùa vụ (`Season`: *Kharif, Rabi, Zaid*), Loại hình tưới (`Irrigation_Type`: *Canal, Drip, Rainfed, Sprinkler*), Nguồn nước (`Water_Source`: *Groundwater, Rainwater, Reservoir, River*), Khu vực (`Region`: *Central, East, North, South, West*).

Ngoài ra, hệ thống tự động tính toán **4 đặc trưng nâng cao** (Feature Engineering) trước khi đưa vào mô hình: `ET_Index`, `Moisture_Deficit`, `Heat_Humidity_Ratio`, `Thermal_Radiation`.
## 3. Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### Yêu cầu môi trường
Python >= 3.9
Thư viện: `streamlit`, `pandas`, `joblib`, `lightgbm`

### Các bước thực hiện

1. **Tải mã nguồn dự án về máy:**
   ```bash
   git clone <URL_REPOSITORY_CỦA_BẠN>
   cd smart-irrigation-lightgbm
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install streamlit pandas joblib lightgbm matplotlib seaborn
   ```

3. **Cấu trúc thư mục dự án chuẩn:**
   ```text
   smart-irrigation-lightgbm/
   ├── models/
   │   └── model_lightgbm.pkl    # File mô hình đã huấn luyện
   ├── app.py                     # Mã nguồn giao diện Streamlit
   └── README.md                  # Tài liệu hướng dẫn
   ```

4. **Khởi chạy ứng dụng Web Streamlit:**
   ```bash
   streamlit run app.py
   ```

5. Trình duyệt sẽ tự động mở giao diện tại địa chỉ: `http://localhost:8501`.

## 4. Hướng Dẫn Sử Dụng Giao Diện Web

1. Xem nhanh bảng dịch thuật ngữ Tiếng Việt tại mục **"Hướng dẫn và Bảng chú thích thuật ngữ Tiếng Việt"** nếu cần.
2. Điều chỉnh các tham số đầu vào thông qua các **thanh trượt** và **danh sách chọn** trong 3 Tab:
   **Tab 1:** Thời Tiết và Môi Trường.
   **Tab 2:** Đất Đai và Cây Trồng.
   **Tab 3:** Tưới Tiêu và Khu Vực.
3. Nhấn nút **"Chạy Dự Đoán Mô Hình LightGBM"**.
4. Kết quả khuyến nghị mức tưới sẽ hiển thị ngay lập tức với màu tương ứng:
   **LOW (Thấp):** Đất đủ ẩm hoặc thời tiết mát mẻ, chưa cần tưới.
   **MEDIUM (Trung Bình):** Cần tưới duy trì ở mức vừa phải.
   **HIGH (Cao):** Đất khô hoặc điều kiện bốc hơi cao, cần kích hoạt máy bơm ngay.