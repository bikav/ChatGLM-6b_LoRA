"""04 分割数据集"""

input_file_path = 'sleep_data_03.txt'
train_file_path = 'sleep_train.txt'
test_file_path = 'sleep_test.txt'
val_file_path = 'sleep_dev.txt'

# 读取文件中的所有行
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 计算分割点
total_lines = len(lines)
train_split_index = int(total_lines * 0.8)  # 训练集的分割点
test_val_split_index = int(total_lines * 0.9)  # 测试集和验证集的分割点

# 分割数据为训练集、测试集和验证集
train_lines = lines[:train_split_index]
test_lines = lines[train_split_index:test_val_split_index]
val_lines = lines[test_val_split_index:]

# 写入训练集数据到文件
with open(train_file_path, 'w', encoding='utf-8') as file:
    file.writelines(train_lines)

# 写入测试集数据到文件
with open(test_file_path, 'w', encoding='utf-8') as file:
    file.writelines(test_lines)

# 写入验证集数据到文件
with open(val_file_path, 'w', encoding='utf-8') as file:
    file.writelines(val_lines)
