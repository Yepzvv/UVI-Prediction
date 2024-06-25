import pandas as pd

data = pd.read_csv('haiyang.csv')

missing_rows = data[data.isnull().any(axis=1)]
zero_rows = data[(data == 0).any(axis=1)]

data.fillna(0, inplace=True)

for i, name in enumerate(data.columns.tolist()):
    print(f"{i+1}  {name}: {data[data[name] != 0.0].shape}")
    print(data[name].unique(), "\n")


columns_to_drop = ['预报时段', '流速（米/秒）', '视程（公里）', '预报时次', '天气状况', 
                   '浪高级别', '预警信号', '写入时间', '预报时间', '主键ID', '天气图标',
                    '气压（百帕）', '最小降雨量（毫米）']
data = data.drop(columns=columns_to_drop)

for i, name in enumerate(data.columns.tolist()):
    print(f"{i+1}  {name}: {data[data[name] != 0.0].shape}")
    print(data[name].unique(), "\n")


valid_values = {'6', '7', '5','4', '3', '8', 
                '10', '2', '9','11', '0', '1', '12'}
data = data[data['阵风（级）'].isin(valid_values)]
data = data[data['风力（级）'].isin(valid_values)]

valid_values = {'东南', '南', '西南', '北', '东', '东北', '西北', '西'}
data = data[data['阵风风向'].isin(valid_values)]
data = data[data['风向'].isin(valid_values)]

valid_values = {'10', '5', '8', '3', '1', '7', '4', '12', '6', '15', '14', '9', '2', '13', '11',
 '20', '1.5', 8.0, 5.0, 10.0, 7.0, 4.0, 3.0, 2.0, 6.0, 20.0, 1.0, 0.2, 15.0, 12.0, 9.0, 11.0, 0.1, 14.0, 13.0, 18.0}
data = data[data['最低能见度（公里）'].isin(valid_values)]

for i, name in enumerate(data.columns.tolist()):
    print(f"{i+1}  {name}: {data[data[name] != 0.0].shape}")
    print(data[name].unique(), "\n")


region_mapping = {
    '深汕海区': 1,
    '三门岛海域': 2,
    '大亚湾': 3,
    '大鹏湾': 4,
    '妈湾蛇口海域': 5,
    '机场西部海域': 6,
    '深圳湾': 7,
    '深圳西部沿海海域': 8,
    '珠江口': 9
}
data['区域名称'] = data['区域名称'].replace(region_mapping)

region_mapping = {
    '东': 1,
    '南': 2,
    '西': 3,
    '北': 4,
    '东南': 5,
    '西南': 6,
    '东北': 7,
    '西北': 8,
    '无': 9
}
data['阵风风向'] = data['阵风风向'].replace(region_mapping)
data['风向'] = data['风向'].replace(region_mapping)

region_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,
    '11': 11,
    '12': 12
}
data['阵风（级）'] = data['阵风（级）'].replace(region_mapping)
data['风力（级）'] = data['风力（级）'].replace(region_mapping)

region_mapping = {
    '1.5': 1.5,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,
    '11': 11,
    '12': 12,
    '13': 13,
    '14': 14,
    '15': 15,
    '20': 20,
}
data['最低能见度（公里）'] = data['最低能见度（公里）'].replace(region_mapping)

for i, name in enumerate(data.columns.tolist()):
    print(f"{i+1}  {name}: {data[data[name] != 0.0].shape}")
    print(data[name].unique(), "\n")


data.to_csv('preprocess1.csv', index=False)
zero_rows = data[(data == 0).any(axis=1)]
