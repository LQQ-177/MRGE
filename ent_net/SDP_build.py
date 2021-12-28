import os
import json


data_path = '../prepro_data'


def path_find(path1, path2):
    #寻找path1和path2路径最深的共同祖先节点
    path1 = path1[::-1]
    path2 = path2[::-1]

    temp = 0
    for l1, l2 in zip(path1, path2):
        if (l1 == l2):
            temp += 1

    #路径不为空，退后一阁
    if temp == min(len(path1),len(path2)):
        temp -= 1

    return path1[temp:], path2[temp:]#返回不同的路径用于LSTM


def gen_data_LSTM(is_training = True, suffix=''):
    #将数据转换为LSTM所需最短路径形式
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    path_len_max = 0

    tree_data = json.load(open(name_prefix + suffix + '_tree.json'))
    ent_data = json.load(open(os.path.join(data_path, name_prefix + suffix + '.json')))

    data = []
    for i in range(len(ent_data)):#分文档处理
        ent_id = {}#存储实例的点
        for j in range(len(ent_data[i]['vertexSet'])):#分实体处理
            layer_min = 100
            id_min = 0

            #print(ent_data[i]['vertexSet'][j][0])
            for k in range(ent_data[i]['vertexSet'][j][0]['pos'][0],ent_data[i]['vertexSet'][j][0]['pos'][1]):#只看第一个实例
                if str(k) in tree_data[i]:
                    if tree_data[i][str(k)]['layer'] < layer_min:
                        layer_min = tree_data[i][str(k)]['layer']
                        id_min = k
            ent_id[j] = id_min
        #print(ent_id)

        item_doc = {}
        for m in range(len(ent_id)):
            head = str(ent_id[m])
            item = {}
            for n in range(len(ent_id)):
                if n == m:
                    continue

                tail = str(ent_id[n])
                path_head, path_tail = path_find(tree_data[i][head]['path'], tree_data[i][tail]['path'])
                item[n] = {'path_head': path_head, 'path_tail':path_tail}
                #print(path_head, path_tail)
                if len(path_head) > path_len_max:
                    path_len_max = len(path_head)
                if len(path_tail) > path_len_max:
                    path_len_max = len(path_tail)
            item_doc[m] = item

        data.append(item_doc)

    #print(data)
    print('path_len_max:',path_len_max)#15/13/9/9

    json.dump(data, open(os.path.join(data_path, name_prefix + suffix + '_tree_LSTM.json'), 'w'))



gen_data_LSTM(is_training = True, suffix = '')
gen_data_LSTM(is_training = False, suffix = '_train')
gen_data_LSTM(is_training = False, suffix = '_dev')
gen_data_LSTM(is_training = False, suffix = '_test')



'''
[{'pos': [1, 6], 'type': 'ORG', 'sent_id': 0, 'name': 'Worker-Peasant Red Guards'}]
[{'pos': [7, 8], 'type': 'ORG', 'sent_id': 0, 'name': 'WPRG'}]
[{'pos': [14, 20], 'type': 'ORG', 'sent_id': 0, 'name': "Workers and Peasants' Red Militia"}]
[{'pos': [21, 22], 'type': 'ORG', 'sent_id': 0, 'name': 'WPRM'}]
[{'pos': [29, 31], 'type': 'LOC', 'sent_id': 0, 'name': 'North Korea'}, {'pos': [40, 42], 'type': 'LOC', 'sent_id': 1, 'name': 'North Korea'}]
[{'pos': [47, 51], 'type': 'TIME', 'sent_id': 2, 'name': 'January 14, 1959'}]
[{'pos': [52, 56], 'type': 'PER', 'sent_id': 2, 'name': 'Kim Il-sung'}]
[{'pos': [61, 64], 'type': 'ORG', 'sent_id': 2, 'name': 'State Affairs Commission'}]
[{'pos': [66, 67], 'type': 'TIME', 'sent_id': 2, 'name': '2016'}]
[{'pos': [67, 70], 'type': 'ORG', 'sent_id': 2, 'name': 'National Defense Commission'}]
[{'pos': [72, 78], 'type': 'ORG', 'sent_id': 2, 'name': "Ministry of People's Armed Forces"}]
[{'pos': [86, 91], 'type': 'ORG', 'sent_id': 2, 'name': "Workers' Party of Korea"}]
[{'pos': [93, 96], 'type': 'ORG', 'sent_id': 2, 'name': 'Military Affairs Department'}]
[{'pos': [155, 156], 'type': 'ORG', 'sent_id': 4, 'name': 'BM-13'}]
[{'pos': [158, 160], 'type': 'MISC', 'sent_id': 4, 'name': 'Ural D-62'}]
[{'pos': [180, 182], 'type': 'NUM', 'sent_id': 5, 'name': '5 million'}]
'''