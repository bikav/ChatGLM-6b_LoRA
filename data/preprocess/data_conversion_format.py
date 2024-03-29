"""02 把json转成txt"""

import json

# 读取原始JSON文件
input_file_path = 'sleep_data_02.json'
output_file_path = 'sleep_data_02.txt'

# 读取并转换数据，然后写入TXT文件
with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w',
                                                                      encoding='utf-8') as output_file:
    # 加载JSON数据
    original_data = json.load(input_file)

    # 遍历每条记录
    for item in original_data:
        # 将每个JSON对象转换成字符串
        json_str = json.dumps(item, ensure_ascii=False)
        # 写入文件，每条记录后加换行符
        output_file.write(json_str + "\n")
