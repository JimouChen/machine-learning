"""
# @Time    :  2020/10/22
# @Author  :  Jimou Chen
"""
import pandas as pd


# 处理数据
def handle_data(data):
    x_test = data.iloc[-1, 1:-1].values
    data = data.iloc[:-1, :]
    return data, x_test


# 计算先验概率,同时返回Yes和No标签的数量
def cal_pre(data):
    num = list(data['PlayTennis'])
    total_len = len(num)
    yes_num = 0
    no_num = 0
    for i in num:
        if i == 'No':
            no_num += 1
        else:
            yes_num += 1
    pre_yes = (yes_num + 1) / (total_len + 2)
    pre_no = (no_num + 1) / (total_len + 2)

    return pre_yes, pre_no, yes_num, no_num


# 计算每个属性的类条件概率
def class_cons(data, x_test, label, nums):
    label_data = list(data['PlayTennis'])
    Outlook_data = list(data['Outlook'])
    Temperature = list(data['Temperature'])
    Humidity = list(data['Humidity'])
    Wind = list(data['Wind'])

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0

    for i in range(len(label_data)):
        if label_data[i] == label and Outlook_data[i] == x_test[0]:
            count1 += 1
        if label_data[i] == label and Temperature[i] == x_test[1]:
            count2 += 1
        if label_data[i] == label and Humidity[i] == x_test[2]:
            count3 += 1
        if label_data[i] == label and Wind[i] == x_test[3]:
            count4 += 1

    res1 = (count1 + 1) / (nums + 2)
    res2 = (count2 + 1) / (nums + 2)
    res3 = (count3 + 1) / (nums + 2)
    res4 = (count4 + 1) / (nums + 2)

    return [res1, res2, res3, res4]


if __name__ == '__main__':
    my_data = pd.read_csv('data.csv')
    data, test_data = handle_data(my_data)

    # 返回先验概率 和 去与不去的数量
    pre_yes, pre_no, yes_num, no_num = cal_pre(data)

    # 每个属性的类条件概率
    yes_res = class_cons(data, test_data, 'Yes', yes_num)
    no_res = class_cons(data, test_data, 'No', no_num)

    # 计算yes和no的概率
    p_yes = pre_yes
    p_no = pre_no
    for each in yes_res:
        p_yes *= each
    for each in no_res:
        p_no *= each

    print('yes的概率是:', p_yes)
    print('no的概率是:', p_no)

    if p_yes > p_no:
        print('经预测，可以去打球！')
    else:
        print('经预测，不可以去打球！')
