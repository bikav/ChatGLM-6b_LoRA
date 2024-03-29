"""01 修改标签"""

import json

# 读取原始数据
with open('sleep_data.json', 'r', encoding='utf-8') as file:
    original_data = json.load(file)

# 数据转换
converted_data = []
for item in original_data:
    new_item = {
        "instruct": item.get("input", ""),
        "query": item.get("instruction", ""),
        "answer": item.get("output", "")
    }
    converted_data.append(new_item)

# 保存转换后的数据
with open('sleep_data_02.json', 'w', encoding='utf-8') as file:
    json.dump(converted_data, file, ensure_ascii=False, indent=4)
