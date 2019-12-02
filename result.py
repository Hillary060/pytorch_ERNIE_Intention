import os
from utils import constant
from sklearn import metrics
import itertools
import warnings

warnings.filterwarnings('ignore')

grained2id = constant.LABEL_TO_ID
# label
labels = eval(open('result/grained/labels').read())

# grained_preds
grained_preds = eval(open('result/grained/preds').read())

# combine predictions of all classifications
res_path = 'result/multi/'

# 多种大类组合的多级分类效果比较
for_multi_list = constant.COARSE_INTO_MULTI
all_kind = list()
for i in range(4):
    all_kind += list(itertools.combinations(for_multi_list, i+1))

# final prediction
best_acc, best_f1 = 0 , 0
for i,multi_list in enumerate(all_kind):
    final_preds = grained_preds[:]
    for i,coarse_name in enumerate(multi_list):
        # coarse_name = '物品携带托运'
        grained2id_in_coarse = constant.GRAINED_ID_IN_COARSE[coarse_name]
        id2grained_in_coarse = {v:k for k,v in grained2id_in_coarse.items()}

        res_dir = os.path.join(res_path, coarse_name)
        index_rela_path = os.path.join(res_dir, 'index_relation')
        preds_path = os.path.join(res_dir, 'preds')

        if os.path.exists(index_rela_path) and os.path.exists(preds_path):
            # index[0~1500] = data index of current coarse
            index_relation = eval(open(index_rela_path).read())

            # test prediction in second level
            preds = eval(open(preds_path).read())
            for k,v in index_relation.items():
                grained_name= id2grained_in_coarse[preds[v]]
                final_preds[k] = grained2id[grained_name]

    # # report
    # report = metrics.classification_report(labels, final_preds, output_dict=False)
    # print(report)

    # evaluation
    acc = metrics.accuracy_score(labels, final_preds)
    f1_score = metrics.f1_score(labels, final_preds, average='macro')

    print(str(multi_list))
    print("acc:"+str(acc)+"\t"+"F1:"+str(f1_score))

    if acc>best_acc:
        best_acc = acc
    if f1_score>best_f1:
        best_f1 = f1_score
print("best acc {} f1 {}".format(best_acc,best_f1))

    # confusion_matrix = metrics.confusion_matrix(labels, final_preds)
    #
    # # print acc f1
    # print("acc:{} f1:{}".format(str(acc),str(f1_score)))
    #
    # # write report to excel
    # del report["accuracy"]
    # del report["macro avg"]
    # del report["weighted avg"]
    # report = {int(float(k)): v for k, v in report.items()}
    # id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
    # new_report = dict()
    # for label_id,label_name in id2label.items():
    #     if label_id not in report:
    #         new_report[label_name] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0, }
    #     else:
    #         report[label_id]['coarse_name'] = constant.GRAINED_TO_COARSE[label_name]
    #         new_report[label_name] = report[label_id]
    #
    # new_report['all'] = {'acc': str(acc), 'recall': recall, 'f1-score': f1_score, 'data_num': len(labels), }
    # df = pd.DataFrame.from_dict(new_report, orient='index')
    # df.to_excel(res_path+'/final_result.xlsx')
