import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, month, year, lag, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os
import time

def create_spark_session():
    """Tạo và trả về một SparkSession"""
    spark = SparkSession.builder \
        .appName("Traffic Flow Prediction với Spark ARIMA") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Đặt mức độ log để giảm thông tin không cần thiết
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_data(spark, train_path, test_path):
    """Đọc dữ liệu train và test từ các tệp CSV"""
    # Đọc dữ liệu train
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)
    
    # Đọc dữ liệu test
    test_df = spark.read.csv(test_path, header=True, inferSchema=True)
    
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Tiền xử lý dữ liệu cho cả tập train và test"""
    # Chuyển đổi cột DateTime thành timestamp
    train_df = train_df.withColumn("DateTime", to_timestamp(col("DateTime")))
    test_df = test_df.withColumn("Datetime", to_timestamp(col("Datetime")))
    
    # Thêm các đặc trưng thời gian cho tập train
    train_df = train_df.withColumn("hour", hour(col("DateTime"))) \
                       .withColumn("dayofweek", dayofweek(col("DateTime"))) \
                       .withColumn("month", month(col("DateTime"))) \
                       .withColumn("year", year(col("DateTime")))
    
    return train_df, test_df

def arima_preprocessing(train_df, num_lags=5):
    """Tạo các tính năng trễ (lag features) cho mô hình ARIMA trong Spark"""
    # Tạo ID duy nhất để sắp xếp dữ liệu
    train_df = train_df.withColumn("id", monotonically_increasing_id())
    
    # Sắp xếp dữ liệu theo thời gian
    train_df = train_df.orderBy("DateTime")
    
    # Định nghĩa cửa sổ trượt để tạo các đặc trưng trễ
    w = Window.orderBy("DateTime")
    
    # Tạo các đặc trưng trễ (lag features) cho mỗi biến
    for i in range(1, num_lags + 1):
        train_df = train_df.withColumn(f"lag_{i}", lag("Vehicles", i).over(w))
    
    # Loại bỏ các hàng có giá trị NaN (các hàng đầu tiên sẽ có NaN do tính năng trễ)
    train_df = train_df.na.drop()
    
    return train_df

def split_data(df, train_ratio=0.8):
    """Chia dữ liệu thành tập huấn luyện và tập kiểm định"""
    # Đếm tổng số bản ghi
    total_count = df.count()
    train_count = int(total_count * train_ratio)
    
    # Chia dữ liệu
    train_data = df.limit(train_count)
    val_data = df.subtract(train_data)
    
    return train_data, val_data

def build_arima_features(df, p, d, q, spark_session=None):
    """Xây dựng đặc trưng cho mô hình ARIMA trong Spark
    
    p: Số độ trễ của thành phần AR (autoregressive)
    d: Số lần lấy sai phân
    q: Số độ trễ của thành phần MA (moving average)
    spark_session: Phiên Spark sử dụng để tạo DataFrame
    """
    # Chuyển DataFrame Spark thành Pandas DataFrame để dễ xử lý
    pandas_df = df.toPandas()
    pandas_df = pandas_df.sort_values('DateTime')
    
    # Thực hiện phép lấy sai phân nếu d > 0
    ts_data = pandas_df['Vehicles']
    for _ in range(d):
        ts_data = ts_data.diff().dropna()
    
    # Tạo các đặc trưng cho AR và MA
    features = pd.DataFrame()
    
    # Thêm các đặc trưng AR
    for i in range(1, p + 1):
        features[f'ar_{i}'] = ts_data.shift(i)
    
    # Thêm các đặc trưng MA (residuals)
    if q > 0:
        # Đối với MA, ta cần tính residuals. Đây là một cách xấp xỉ:
        # Dùng mô hình AR(p) để dự đoán và tính residuals
        ar_model = ARIMA(ts_data.dropna().values, order=(p, 0, 0))
        ar_result = ar_model.fit()
        residuals = ar_result.resid
        
        # Thêm residuals như các đặc trưng MA
        residuals_df = pd.DataFrame(residuals, columns=['resid'])
        for i in range(1, q + 1):
            features[f'ma_{i}'] = residuals_df['resid'].shift(i)
    
    # Thêm cột 'target' là giá trị hiện tại của chuỗi thời gian
    features['target'] = ts_data
    
    # Loại bỏ dữ liệu NaN
    features = features.dropna()
    
    # Chuyển lại thành Spark DataFrame
    # Sử dụng spark_session nếu được cung cấp, nếu không thì tìm biến spark từ module hiện tại
    if spark_session is None:
        # Lấy biến spark từ module gọi hàm này
        import sys
        spark_session = sys.modules['__main__'].spark
    
    spark_features = spark_session.createDataFrame(features)
    
    return spark_features, d

def train_spark_arima(features_df, p, d, q):
    """Huấn luyện mô hình ARIMA sử dụng Spark ML
    
    Thực chất đây là một cách xấp xỉ ARIMA sử dụng các mô hình hồi quy tuyến tính
    """
    # Chuẩn bị đặc trưng đầu vào
    feature_cols = [col for col in features_df.columns if col != 'target']
    
    # Tạo vector đặc trưng
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(features_df)
    
    # Huấn luyện mô hình hồi quy tuyến tính
    lr = LinearRegression(featuresCol="features", labelCol="target", maxIter=100, regParam=0.0)
    model = lr.fit(data)
    
    return model, feature_cols

def predict_with_spark_arima(model, features_df, feature_cols, d, original_series=None):
    """Dự đoán sử dụng mô hình Spark ARIMA"""
    # Tạo vector đặc trưng
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(features_df)
    
    # Dự đoán
    predictions = model.transform(data)
    
    # Nếu đã thực hiện sai phân (d > 0), cần tích phân ngược lại
    if d > 0 and original_series is not None:
        # Chuyển về pandas để dễ tính toán
        pred_df = predictions.select("prediction").toPandas()
        original_df = original_series.toPandas()
        
        # Tích phân ngược
        for _ in range(d):
            # Lấy giá trị cuối cùng từ chuỗi gốc
            last_value = original_df['Vehicles'].iloc[-1]
            # Cộng dồn sai phân để lấy lại giá trị gốc
            pred_df['prediction'] = pred_df['prediction'].cumsum() + last_value
        
        # Chuyển lại thành Spark DataFrame
        # Sử dụng biến spark từ module gọi hàm này
        import sys
        spark_session = sys.modules['__main__'].spark
        predictions = spark_session.createDataFrame(pred_df)
    
    return predictions

def evaluate_predictions(predictions_df, actual_df):
    """Đánh giá dự đoán bằng MAE (Mean Absolute Error)"""
    # Chuyển cả hai DataFrame về pandas để dễ tính toán
    pred = predictions_df.select("prediction").toPandas()
    actual = actual_df.select("Vehicles").toPandas()
    
    # Đảm bảo cùng độ dài
    min_len = min(len(pred), len(actual))
    pred = pred.iloc[:min_len]
    actual = actual.iloc[:min_len]
    
    # Tính MAE
    mae = np.mean(np.abs(actual['Vehicles'].values - pred['prediction'].values))
    
    return mae

def save_predictions(predictions, test_df, output_path="predictions.csv"):
    """Lưu kết quả dự đoán vào file CSV"""
    # Chuyển kết quả dự đoán về pandas
    pred_df = predictions.select("prediction").toPandas()
    test_pandas = test_df.toPandas()
    
    # Đảm bảo cùng độ dài
    min_len = min(len(pred_df), len(test_pandas))
    pred_df = pred_df.iloc[:min_len]
    test_pandas = test_pandas.iloc[:min_len]
    
    # Thêm cột Vehicles với giá trị dự đoán
    test_pandas['Vehicles'] = pred_df['prediction'].values
    
    # Lưu kết quả
    test_pandas[['ID', 'Vehicles']].to_csv(output_path, index=False)

def grid_search_arima(train_data, val_data, p_values, d_values, q_values):
    """Tìm kiếm tham số tối ưu cho mô hình ARIMA"""
    best_mae = float('inf')
    best_params = None
    best_model = None
    best_feature_cols = None
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    # Xây dựng đặc trưng ARIMA
                    train_features, actual_d = build_arima_features(train_data, p, d, q)
                    
                    # Huấn luyện mô hình
                    model, feature_cols = train_spark_arima(train_features, p, d, q)
                    
                    # Xây dựng đặc trưng cho tập validation
                    val_features, _ = build_arima_features(val_data, p, d, q)
                    
                    # Dự đoán trên tập validation
                    predictions = predict_with_spark_arima(model, val_features, feature_cols, actual_d, train_data)
                    
                    # Đánh giá
                    val_actual = val_data.select("Vehicles")
                    mae = evaluate_predictions(predictions, val_actual)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_params = (p, d, q)
                        best_model = model
                        best_feature_cols = feature_cols
                
                except Exception as e:
                    pass
    
    return best_model, best_params, best_feature_cols, best_mae

def main():
    # Đường dẫn đến dữ liệu
    train_path = "train.csv"
    test_path = "test.csv"
    
    global spark
    # Khởi tạo Spark Session
    spark = create_spark_session()
    
    try:
        # Đọc dữ liệu
        train_df, test_df = load_data(spark, train_path, test_path)
        
        # Tiền xử lý dữ liệu
        train_df, test_df = preprocess_data(train_df, test_df)
        
        # Tạo các đặc trưng trễ (lag features) cho ARIMA
        train_df_with_lags = arima_preprocessing(train_df, num_lags=10)
        
        # Chia dữ liệu
        train_data, val_data = split_data(train_df_with_lags)
        
        # Grid search để tìm tham số tối ưu
        p_values = [1, 2, 3]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        
        best_model, best_params, best_feature_cols, best_mae = grid_search_arima(
            train_data, val_data, p_values, d_values, q_values)
        
        # Huấn luyện lại mô hình trên toàn bộ dữ liệu huấn luyện
        all_features, actual_d = build_arima_features(train_df, 
                                                     best_params[0], 
                                                     best_params[1], 
                                                     best_params[2])
        
        final_model, final_feature_cols = train_spark_arima(all_features, 
                                                          best_params[0], 
                                                          best_params[1], 
                                                          best_params[2])
        
        # Chuẩn bị dữ liệu test cho dự đoán
        # Đây là bước phức tạp vì dữ liệu test không có cột Vehicles
        # Chúng ta cần dự đoán tuần tự và cập nhật các đặc trưng
        
        # Để đơn giản, chúng ta sẽ sử dụng Pandas để triển khai dự đoán
        test_pandas = test_df.toPandas()
        train_pandas = train_df.toPandas().sort_values('DateTime')
        
        # Chuẩn bị mô hình ARIMA từ statsmodels để dự đoán
        arima_model = ARIMA(train_pandas['Vehicles'].values, 
                          order=(best_params[0], best_params[1], best_params[2]))
        arima_fit = arima_model.fit()
        
        # Dự đoán
        forecast = arima_fit.forecast(steps=len(test_pandas))
        
        # Chuyển dự đoán thành DataFrame
        test_pandas['Vehicles'] = forecast
        
        # Lưu kết quả
        test_pandas[['ID', 'Vehicles']].to_csv('spark_arima_predictions.csv', index=False)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        plt.plot(train_pandas['DateTime'], train_pandas['Vehicles'], label='Dữ liệu huấn luyện')
        plt.plot(test_pandas['Datetime'], test_pandas['Vehicles'], label='Dự đoán', color='red')
        plt.title('Dự đoán lưu lượng xe với Spark ARIMA')
        plt.xlabel('Thời gian')
        plt.ylabel('Số lượng xe')
        plt.legend()
        plt.savefig('spark_arima_prediction_plot.png')
        
    finally:
        # Đóng Spark Session
        spark.stop()

if __name__ == "__main__":
    main() 