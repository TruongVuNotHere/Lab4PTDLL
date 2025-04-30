import pandas as pd
import time
import socket
import json
import os
from datetime import datetime
import sys
import math

def load_data(file_path):
    """Đọc dữ liệu từ tập CSV sử dụng pandas"""
    print(f"Đang đọc dữ liệu từ {file_path}...")
    df = pd.read_csv(file_path)
    
    # Chuyển đổi cột DateTime thành timestamp nếu cần
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    print(f"Đã đọc xong {len(df)} bản ghi")
    return df

def send_data(df, target_host="localhost", target_port=9999, batch_size=1000, duration_per_batch=5.0):
    """Gửi dữ liệu từ DataFrame đến server qua socket
    
    Tham số:
    - df: DataFrame chứa dữ liệu
    - target_host: Địa chỉ server
    - target_port: Cổng kết nối
    - batch_size: Số bản ghi gửi trong mỗi batch
    - duration_per_batch: Thời gian dự kiến cho mỗi batch (giây)
    """
    # Sắp xếp dữ liệu theo thời gian nếu có cột DateTime
    if 'DateTime' in df.columns:
        df = df.sort_values('DateTime')
    
    # Tính toán số batch cần gửi
    total_records = len(df)
    batches = math.ceil(total_records / batch_size)
    
    # Kết nối đến socket server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (target_host, target_port)
        print(f"Đang kết nối đến {server_address}...")
        sock.connect(server_address)
        print(f"Đã kết nối thành công đến {server_address}")
        
        # Gửi dữ liệu
        try:
            records_sent = 0
            start_time = time.time()
            
            print(f"Bắt đầu gửi {total_records} bản ghi trong {batches} batch, mỗi batch {batch_size} bản ghi")
            print(f"Thời gian dự kiến cho mỗi batch: {duration_per_batch} giây")
            
            for batch_idx in range(batches):
                batch_start_time = time.time()
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_records)
                batch_size_actual = batch_end - batch_start
                
                batch_data = []
                for i in range(batch_start, batch_end):
                    row = df.iloc[i]
                    data = {
                        "ID": str(row["ID"]),
                        "DateTime": row["DateTime"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row["DateTime"], datetime) else str(row["DateTime"]),
                        "Junction": int(row["Junction"]),
                        "Vehicles": int(row["Vehicles"])
                    }
                    batch_data.append(data)
                
                # Chuyển đổi toàn bộ batch thành JSON và gửi
                json_data = "\n".join([json.dumps(record) for record in batch_data]) + "\n"
                sock.sendall(json_data.encode())
                
                records_sent += batch_size_actual
                
                # Cập nhật tiến trình
                elapsed_time = time.time() - start_time
                progress = records_sent / total_records * 100
                speed = records_sent / elapsed_time if elapsed_time > 0 else 0
                remaining = (total_records - records_sent) / speed if speed > 0 else 0
                
                sys.stdout.write(f"\rBatch {batch_idx+1}/{batches}: Đã gửi {records_sent}/{total_records} bản ghi ({progress:.1f}%) | " +
                                f"Tốc độ: {speed:.1f} bản ghi/giây | Còn lại: {remaining:.1f} giây")
                sys.stdout.flush()
                
                # Điều chỉnh thời gian nghỉ giữa các batch để đạt được tốc độ mong muốn
                batch_elapsed = time.time() - batch_start_time
                sleep_time = max(0, duration_per_batch - batch_elapsed)
                if sleep_time > 0 and batch_idx < batches - 1:  # Không cần nghỉ sau batch cuối cùng
                    time.sleep(sleep_time)
            
            total_time = time.time() - start_time
            print(f"\nĐã hoàn thành gửi {records_sent} bản ghi trong {total_time:.2f} giây ({records_sent/total_time:.1f} bản ghi/giây)")
            
        except Exception as e:
            print(f"\nLỗi khi gửi dữ liệu: {e}")
        
        finally:
            print("Đóng kết nối...")
            sock.close()
            
    except ConnectionRefusedError:
        print("Không thể kết nối đến server. Hãy đảm bảo server đang chạy.")
    except Exception as e:
        print(f"Lỗi khi thiết lập kết nối: {e}")

def main():
    # Đường dẫn đến file dữ liệu
    data_path = "train.csv"
    
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file {data_path}!")
        return
    
    # Mặc định: gửi toàn bộ dữ liệu, mỗi lần 1000 bản ghi
    host = "localhost"
    port = 9999
    batch_size = 1000     # Số bản ghi gửi trong mỗi batch
    duration_per_batch = 5.0  # Thời gian cho mỗi batch (giây)
    
    # Kiểm tra tham số dòng lệnh nếu có
    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--help" or sys.argv[i] == "-h":
                print("Sử dụng: python data_sender.py [--batch-size SIZE] [--duration-per-batch SECONDS] [--host HOST] [--port PORT]")
                return
            elif sys.argv[i] == "--batch-size" and i+1 < len(sys.argv):
                batch_size = int(sys.argv[i+1])
                i += 2
            elif sys.argv[i] == "--duration-per-batch" and i+1 < len(sys.argv):
                duration_per_batch = float(sys.argv[i+1])
                i += 2
            elif sys.argv[i] == "--host" and i+1 < len(sys.argv):
                host = sys.argv[i+1]
                i += 2
            elif sys.argv[i] == "--port" and i+1 < len(sys.argv):
                port = int(sys.argv[i+1])
                i += 2
            else:
                i += 1
    
    try:
        # Đọc dữ liệu
        df = load_data(data_path)
        
        # Gửi dữ liệu
        print(f"Cấu hình: host={host}, port={port}, batch_size={batch_size}, duration_per_batch={duration_per_batch}")
        send_data(df, target_host=host, target_port=port, 
                 batch_size=batch_size, duration_per_batch=duration_per_batch)
        
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    print("=== SERVER GỬI DỮ LIỆU LƯU LƯỢNG GIAO THÔNG ===")
    main() 