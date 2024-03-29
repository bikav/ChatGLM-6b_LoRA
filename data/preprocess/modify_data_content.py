"""03 修改instruct字段的内容"""

import json

# 要添加到instruct字段的文本
additional_text = "你现在是一个医生，病人会向你提出一些睡眠上的问题，你要帮忙解答。问题："

# 读取原始数据集文件
with open('sleep_data_02.txt', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 修改每条数据中的instruct字段
for item in data:
    item['instruct'] = additional_text

# 将修改后的数据写回到新文件中
with open('sleep_data_03.txt', 'w', encoding='utf-8') as file:
    for item in data:
        json.dump(item, file, ensure_ascii=False)
        file.write('\n')
