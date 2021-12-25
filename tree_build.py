import os
import json

java_path = "/opt/jdk1.8.0_231/bin/java"
os.environ['JAVAHOME'] = java_path
# Dependency Tree
from nltk.parse.stanford import StanfordDependencyParser
dep_parser=StanfordDependencyParser("/home/kgcode/data_2/lqq/DocRED-master/jars/stanford-parser.jar",
                                    "/home/kgcode/data_2/lqq/DocRED-master/jars/stanford-parser-4.2.0-models.jar",
                                    "/home/kgcode/data_2/lqq/DocRED-master/jars/englishPCFG.ser.gz")
'''
os.environ['STANFORD_PARSER'] = 'D:/Desktop/stanford-parser-full-2020-11-17/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'D:/Desktop/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'

# Dependency Tree
from nltk.parse.stanford import StanfordDependencyParser
dep_parser=StanfordDependencyParser(model_path="D:/Desktop/stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
'''

data_path = '../prepro_data'

train_annotated_file_name = os.path.join(data_path, 'dev_train.json')
train_distant_file_name = os.path.join(data_path, 'train.json')
dev_file_name = os.path.join(data_path, 'dev_dev.json')
test_file_name = os.path.join(data_path, 'dev_test.json')


#将文档的所有数组合成一颗森林
dependency_type = {}
def tree_build(data_file_name, is_training = True, suffix=''):
    ori_data = json.load(open(data_file_name))
    data = []

    print('doc_num:', len(ori_data))
    for i in range(len(ori_data)):#分文档处理
        print('doc', i)

        sents = ori_data[i]['sents']
        Ls = ori_data[i]['Ls']#句子起始位

        doc_item = {}

        for j in range(len(sents)):#分句子处理
            print(j)
            parse_tree = dep_parser.parse(sents[j])
            for trees in parse_tree:
                tree = trees
            #print(tree)

            for k in tree.nodes:
                if k == 0:
                    continue
                item = {}

                item['rel'] = tree.nodes[k]['rel']
                if tree.nodes[k]['head'] == 0:
                    item['head'] = -1#对应句子下标，虚拟节点为-1
                else:
                    item['head'] = tree.nodes[k]['head'] - 1 + Ls[j]
                #item['word'] = tree.nodes[k]['word']


                doc_item[tree.nodes[k]['address'] - 1 + Ls[j]] = item

                if item['rel'] not in dependency_type:
                    dependency_type[item['rel']] = 0
                dependency_type[item['rel']] += 1

        data.append(doc_item)

    #print(data)
    #print(denpendency_type)

    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    json.dump(data, open(name_prefix + suffix + '_tree.json', 'w'))


tree_build(train_distant_file_name, is_training = True, suffix = '')
tree_build(train_annotated_file_name, is_training = False, suffix = '_train')
tree_build(dev_file_name, is_training = False, suffix = '_dev')
tree_build(test_file_name, is_training = False, suffix = '_test')


dependency_map = dict((i, j) for i, j in zip(dependency_type.keys(), range(len(dependency_type))))
#print(denpendency_map)
json.dump(dependency_map, open('dependency_map.json', 'w'))


#依存关系对应id
dep_map = json.load(open('dependency_map.json'))
print("dependency_num:",len(dep_map))

def dep2id(data_file_name):
    data = json.load(open(data_file_name))

    for i in range(len(data)):
        for j in data[i].keys():
            data[i][j]['rel'] = dep_map[data[i][j]['rel']]
    #print(data)
    json.dump(data, open(data_file_name, 'w'))


dep2id('train_tree.json')
dep2id('dev_train_tree.json')
dep2id('dev_dev_tree.json')
dep2id('dev_test_tree.json')

'''
数据错误：
"\u00a0"
'''
