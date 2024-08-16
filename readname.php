<?php
// Đọc tham số từ dòng lệnh hoặc sử dụng giá trị mặc định
$sourcePath = $argv[1] ?? './text/';
$outputFile = $argv[2] ?? './file_names.txt';

// Xóa tệp đầu ra nếu có
unlink($outputFile);

// Lấy danh sách tên các tệp PHP và lưu vào tệp đầu ra
file_put_contents($outputFile, implode("\n", getPhpFileNames($sourcePath)));

// Hàm lấy tên các tệp PHP
function getPhpFileNames($path) {
    $phpFiles = array();

    // Đọc các tệp trong thư mục
    $files = scandir($path);

    foreach ($files as $file) {
        // Bỏ qua các thư mục hiện tại và thư mục mẹ
        if ($file !== '.' && $file !== '..') {
            // Kiểm tra nếu là tệp PHP
            if (pathinfo($file, PATHINFO_EXTENSION) === 'php') {
                $phpFiles[] = $file;
            }
        }
    }

    return $phpFiles;
}
?>
