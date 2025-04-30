import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import socket
import json
from datetime import datetime

# Import các hàm từ spark_arima.py
from spark_arima import (
    create_spark_session,
    arima_preprocessing,
    build_arima_features,
    train_spark_arima,
    predict_with_spark_arima,
    evaluate_predictions,
    split_data
)

# Biến spark toàn cục để sử dụng trong toàn bộ file
spark = None

def define_schema():
    """Định nghĩa schema cho dữ liệu"""
    schema = StructType([
        StructField("ID", StringType(), True),
        StructField("DateTime", StringType(), True),
        StructField("Junction", IntegerType(), True),
        StructField("Vehicles", IntegerType(), True)
    ])
    return schema

def collect_data_socket(port=9999):
    """Thu thập dữ liệu từ socket và CHỈ trả về sau khi đã nhận toàn bộ dữ liệu
    
    Tham số:
    - port: Cổng để nhận dữ liệu
    """
    # Tạo một socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind socket đến địa chỉ và cổng
    server_address = ('localhost', port)
    print(f"Bắt đầu server tại {server_address}")
    server_socket.bind(server_address)
    
    # Lắng nghe kết nối
    server_socket.listen(1)
    print(f"Đang lắng nghe kết nối tại cổng {port}...")
    
    # Chấp nhận kết nối
    connection, client_address = server_socket.accept()
    print(f"Nhận kết nối từ {client_address}")
    
    # Thu thập dữ liệu
    collected_data = []
    total_records = 0
    start_time = time.time()
    last_update_time = start_time
    connection_active = True
    
    try:
        # Nhận dữ liệu
        buffer = ""
        while connection_active:
            data = connection.recv(4096)
            
            if not data:
                # Kết nối đã đóng - CHỈ lúc này mới kết thúc thu thập dữ liệu
                print(f"\nKết nối đã đóng từ phía client. Đã nhận toàn bộ dữ liệu: {total_records} bản ghi.")
                connection_active = False
                break
            
            # Xử lý dữ liệu nhận được
            buffer += data.decode()
            
            # Xử lý từng dòng JSON trong buffer
            while '\n' in buffer:
                # Tách dòng đầu tiên từ buffer
                line, buffer = buffer.split('\n', 1)
                
                try:
                    if line.strip():
                        record = json.loads(line)
                        collected_data.append(record)
                        total_records += 1
                        
                        # Hiển thị tiến trình mỗi 100 bản ghi hoặc sau mỗi giây
                        current_time = time.time()
                        if total_records % 100 == 0 or (current_time - last_update_time) >= 1.0:
                            elapsed = current_time - start_time
                            speed = total_records / elapsed if elapsed > 0 else 0
                            last_update_time = current_time
                            
                            # Hiển thị thông tin rõ ràng hơn
                            print(f"\rĐã nhận {total_records} bản ghi - Tốc độ: {speed:.1f} bản ghi/giây - Thời gian: {elapsed:.1f}s", end="")
                            
                except Exception as e:
                    print(f"\nLỗi khi xử lý dòng JSON: {e}")
    
    except KeyboardInterrupt:
        print("\nDừng nhận dữ liệu theo yêu cầu người dùng...")
    except Exception as e:
        print(f"\nLỗi khi nhận dữ liệu: {e}")
    finally:
        # Đóng kết nối
        connection.close()
        server_socket.close()
        print("\nĐã đóng kết nối socket")
    
    print(f"\nKết quả thu thập dữ liệu: {total_records} bản ghi trong {time.time() - start_time:.2f} giây")
    
    # Chỉ trả về dữ liệu khi đã nhận toàn bộ (kết nối đóng)
    if not connection_active and total_records > 0:
        print("Đã nhận xong toàn bộ dữ liệu, bắt đầu xử lý...")
        return collected_data
    else:
        print("Quá trình thu thập dữ liệu bị gián đoạn, không nhận được toàn bộ dữ liệu!")
        return None

def prepare_data_for_model(collected_data):
    """Chuẩn bị dữ liệu cho mô hình Spark ARIMA"""
    if not collected_data:
        print("Không có dữ liệu để xử lý!")
        return None
        
    # Chuyển đổi dữ liệu thành DataFrame
    df = pd.DataFrame(collected_data)
    
    # Chuyển đổi cột DateTime thành timestamp
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Sắp xếp dữ liệu theo thời gian
    df = df.sort_values('DateTime')
    
    print(f"\n=== THÔNG TIN DỮ LIỆU ===")
    print(f"Số lượng bản ghi: {len(df)}")
    print(f"Khoảng thời gian: {df['DateTime'].min()} đến {df['DateTime'].max()}")
    print(f"Số lượng giao lộ (Junction): {df['Junction'].nunique()}")
    print(f"Giá trị Vehicles: Min={df['Vehicles'].min()}, Trung bình={df['Vehicles'].mean():.2f}, Max={df['Vehicles'].max()}")
    
    return df

def train_with_spark_arima(df, p=2, d=1, q=1):
    """Huấn luyện mô hình sử dụng Spark ARIMA"""
    print("\n=== HUẤN LUYỆN MÔ HÌNH SPARK ARIMA ===")
    
    try:
        # Khởi tạo Spark Session và đặt nó vào biến toàn cục
        global spark
        spark = create_spark_session()
        
        # Chuyển pandas DataFrame sang Spark DataFrame
        spark_df = spark.createDataFrame(df)
        
        # Thêm các đặc trưng trễ (lag features)
        spark_df_with_lags = arima_preprocessing(spark_df, num_lags=10)
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm định
        train_data, val_data = split_data(spark_df_with_lags, train_ratio=0.8)
        
        # Xây dựng đặc trưng ARIMA
        train_features, actual_d = build_arima_features(train_data, p, d, q)
        
        # Huấn luyện mô hình
        model, feature_cols = train_spark_arima(train_features, p, d, q)
        
        # Xây dựng đặc trưng cho tập validation
        val_features, _ = build_arima_features(val_data, p, d, q)
        
        # Dự đoán và đánh giá
        predictions = predict_with_spark_arima(model, val_features, feature_cols, actual_d, train_data)
        val_actual = val_data.select("Vehicles")
        mae = evaluate_predictions(predictions, val_actual)
        
        # Dự đoán 24 giờ tiếp theo
        print("\n=== DỰ ĐOÁN 24 GIỜ TIẾP THEO ===")
        
        # Chuyển đổi lại thành pandas để dễ thao tác
        pandas_df = df.copy()
        
        # Lấy thời gian cuối cùng trong dữ liệu
        last_datetime = pandas_df['DateTime'].max()
        
        # Chuẩn bị mô hình ARIMA từ statsmodels (đơn giản hơn cho dự đoán nhiều bước)
        from statsmodels.tsa.arima.model import ARIMA
        
        # Huấn luyện mô hình ARIMA với statsmodels
        arima_model = ARIMA(pandas_df['Vehicles'].values, order=(p, d, q))
        arima_fit = arima_model.fit()
        
        # Dự đoán 24 giờ tiếp theo
        forecast_steps = 24
        forecast = arima_fit.forecast(steps=forecast_steps)
        
        # Tạo dữ liệu thời gian cho dự đoán
        forecast_dates = pd.date_range(start=last_datetime, periods=forecast_steps+1, freq='H')[1:]
        
        # Tạo DataFrame dự đoán
        forecast_df = pd.DataFrame({
            'DateTime': forecast_dates,
            'Forecast': forecast
        })
        
        # Hiển thị kết quả dự đoán
        print(forecast_df)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        plt.plot(pandas_df['DateTime'], pandas_df['Vehicles'], label='Dữ liệu thực tế')
        plt.plot(forecast_df['DateTime'], forecast_df['Forecast'], label='Dự đoán', color='red')
        plt.title('Dự đoán lưu lượng phương tiện với mô hình Spark ARIMA')
        plt.xlabel('Thời gian')
        plt.ylabel('Số lượng phương tiện')
        plt.legend()
        
        # Lưu biểu đồ
        output_dir = "predictions"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{output_dir}/spark_forecast_plot_{timestamp}.png')
        print(f"Đã lưu biểu đồ dự đoán vào file '{output_dir}/spark_forecast_plot_{timestamp}.png'")
        
        # Lưu kết quả dự đoán
        forecast_df.to_csv(f'{output_dir}/spark_forecast_{timestamp}.csv', index=False)
        print(f"Đã lưu kết quả dự đoán vào file '{output_dir}/spark_forecast_{timestamp}.csv'")
        
        # Hiển thị MAE
        print(f"\n=== ĐÁNH GIÁ MÔ HÌNH ===")
        print(f"MAE (Mean Absolute Error): {mae:.2f}")
        
        return model, feature_cols, mae
        
    except Exception as e:
        print(f"Lỗi khi huấn luyện mô hình Spark ARIMA: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    finally:
        # Đóng Spark Session (nếu đã khởi tạo)
        if spark:
            spark.stop()
            print("Đã đóng Spark Session")

def main():
    try:
        # Cấu hình mặc định
        port = 9999
        
        # Thu thập dữ liệu - CHỈ trả về kết quả sau khi đã nhận toàn bộ dữ liệu
        print(f"Bắt đầu nhận dữ liệu. Chờ đến khi kết nối đóng để xác nhận đã nhận hết toàn bộ dữ liệu...")
        collected_data = collect_data_socket(port=port)
        
        if collected_data:
            print(f"Đã nhận toàn bộ dữ liệu ({len(collected_data)} bản ghi), bắt đầu huấn luyện mô hình...")
            
            # Chuẩn bị dữ liệu
            df = prepare_data_for_model(collected_data)
            
            if df is not None:
                # Huấn luyện mô hình Spark ARIMA
                p, d, q = 2, 1, 1  # Tham số mặc định cho mô hình ARIMA
                model, feature_cols, mae = train_with_spark_arima(df, p, d, q)
                
                if model:
                    print("\nHuấn luyện mô hình thành công!")
        else:
            print("Không nhận được toàn bộ dữ liệu, không thể huấn luyện mô hình.")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== SERVER NHẬN DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH ARIMA VỚI SPARK ===")
    print("Lưu ý: Server sẽ CHỈ huấn luyện mô hình sau khi nhận được TOÀN BỘ dữ liệu!")
    main() 