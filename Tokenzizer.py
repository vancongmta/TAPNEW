import subprocess

def run_php_script(php_file, *args):
    try:
        # Tạo lệnh để chạy tệp PHP với các tham số
        command = ['php', php_file] + list(args)
        
        # Thực thi lệnh PHP
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # In ra kết quả của lệnh PHP
        print(f"Kết quả của {php_file} (STDOUT):", result.stdout)
        if result.stderr:
            print(f"Lỗi của {php_file} (STDERR):", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Đã xảy ra lỗi khi chạy {php_file}: {e}")

# Thay đổi các tham số ở đây
php_file1 = 'Tokenizer.php'
safe_path = './text/'
unsafe_path = './unsafenew/'
safe_token_file = './safe.txt'
unsafe_token_file = './unsafenew.txt'

php_file2 = 'readname.php'
output_file = './file_names.txt'

# Chạy tệp PHP đầu tiên với các tham số
run_php_script(php_file1, safe_path, unsafe_path, safe_token_file, unsafe_token_file)

# Chạy tệp PHP thứ hai với các tham số
run_php_script(php_file2, safe_path, output_file)
