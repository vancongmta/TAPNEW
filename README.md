# TAP
TAP
DEMO phát hiện lỗ hổng bảo mật trong source code php
# KIỂM TRA LỖ HỔNG TRONG FILE text:
# Chuyển đổi các mã nguồn trong file text thành token:
python Tokenzizer.py 

kết quả được lưu trong file safe.txt ( mỗi mã nguồn tương ứng với một dòng token) , tên file được chuyển đổi được lưu trong file file_names.txt
![image](https://github.com/user-attachments/assets/507e37b2-e138-4d1e-a472-a1bffc17a067)
![image](https://github.com/user-attachments/assets/6651bdf9-b129-4cb3-acae-57394e4960d8)

# Chạy kết quả kiểm tra
python demo2.py

Kết quả các nhãn dán được lưu trong trong file predicted_labels.txt 
![image](https://github.com/user-attachments/assets/22a4acf0-0361-466e-a103-41a44182e77e)
Mỗi mã CWE là mã định danh sẽ tương ứng với các lỗ hổng. " safe" là xác định là mã nguồn an toàn
Tỷ lệ phần trăm xác định nhãn dán được lưu trong file predicted_labels_with_probabilities.txt




