import pandas as pd
import os
from utils import constant
from sklearn import metrics

# test labels
test_label = pd.read_csv('dataset/test.tsv','\t',header=None)
test_label = list(test_label[0])

# combine predictions of all classifications
res_path = './multi_test_res'
test_pred = {}
for coarse_name in constant.COARSE_TO_ID:
    index_rela_path = res_path+'/index_relation_'+coarse_name
    if os.path.exists(index_rela_path) and os.path.exists(res_path+'/pred_'+coarse_name) :
        index_relation = eval(open(index_rela_path).read())
        pred = eval(open(res_path+'/pred_'+coarse_name).read())
        for k,v in index_relation.items():
            test_pred[k] = pred[v]

for i in range(len(test_label)):
    if i not in test_pred:
        test_pred[i] = -1

# test predictions
test_pred = list(test_pred.values())

report = metrics.classification_report(test_label, test_pred, output_dict=True)

acc = metrics.accuracy_score(test_label,test_pred)
recall = metrics.recall_score(test_label, test_pred, average='macro')
f1_score = metrics.f1_score(test_label, test_pred, average='macro')
confusion_matrix = metrics.confusion_matrix(test_label, test_pred)

# print acc f1
print("acc:{} f1:{}".format(str(acc),str(f1_score)))

# write report to excel
del report["accuracy"]
del report["macro avg"]
del report["weighted avg"]
report = {int(k): v for k, v in report.items()}
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
new_report = dict()
for label_id,label_name in id2label.items():
    if label_id not in report:
        new_report[label_name] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0, }
    else:
        report[label_id]['coarse_name'] = constant.GRAINED_TO_COARSE[label_name]
        new_report[label_name] = report[label_id]

new_report['all'] = {'acc': str(acc), 'recall': recall, 'f1-score': f1_score, 'data_num': len(test_label), }
df = pd.DataFrame.from_dict(new_report, orient='index')
df.to_excel(res_path+'/multi_result.xlsx')