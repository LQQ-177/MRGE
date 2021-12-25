import os
import json

data_path = '../prepro_data'

train_annotated_file_name = 'dev_train_tree.json'
train_distant_file_name = 'train_tree.json'
dev_file_name = 'dev_dev_tree.json'
test_file_name = 'dev_test_tree.json'


def get_path(tree, node, node_t=-1):
    #寻找node到node_t的路径，node_t默认为根节点
    path = []
    path.append(node)
    while (node != node_t):
        node = tree[str(node)]['head']
        path.append(node)
    return path

def get_layer(data_file_name, is_training = True, suffix=''):
    #生成节点到根节点的路径，及其深度
    data = json.load(open(data_file_name))
    for i in range(len(data)):
        for j in data[i].keys():
            path = get_path(data[i], int(j))
            #print(path)
            layer_num = len(path) - 1

            data[i][j]['path'] = path
            data[i][j]['layer'] = layer_num

    #print(data)
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    json.dump(data, open(name_prefix + suffix + '_tree.json', 'w'))


get_layer(train_distant_file_name, is_training = True, suffix = '')
get_layer(train_annotated_file_name, is_training = False, suffix = '_train')
get_layer(dev_file_name, is_training = False, suffix = '_dev')
get_layer(test_file_name, is_training = False, suffix = '_test')




def gen_data_RNN(data_file_name, is_training = True, suffix=''):
    #将数据转换为RNN分层处理形式
    ori_data = json.load(open(data_file_name))

    data = []
    for i in range(len(ori_data)):#分文档处理
        doc_item= {}
        for j in ori_data[i].keys():#分word处理
            item = {}
            layer = ori_data[i][j]['layer']
            item['rel'] = ori_data[i][j]['rel']
            item['head'] = ori_data[i][j]['head']
            item['tail'] = int(j)

            if layer not in doc_item:
                doc_item[layer] = list()
            doc_item[layer].append(item)

        #print(doc_item)

        data.append(doc_item)

    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    json.dump(data, open(os.path.join(data_path, name_prefix + suffix + '_tree_RNN.json'), 'w'))


gen_data_RNN(train_distant_file_name, is_training = True, suffix = '')
gen_data_RNN(train_annotated_file_name, is_training = False, suffix = '_train')
gen_data_RNN(dev_file_name, is_training = False, suffix = '_dev')
gen_data_RNN(test_file_name, is_training = False, suffix = '_test')
