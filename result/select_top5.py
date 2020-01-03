from utils import constant
import numpy as np
import pandas as pd
from sklearn import metrics


def softmax(pred):
    logits = [item[1] for i, item in enumerate(pred)]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=0)
    pred_softmax = [[item[0], probs[i]] for i, item in enumerate(pred)]
    return pred_softmax


# label id and relations
coarse2id = constant.COARSE_TO_ID
grained2id = constant.LABEL_TO_ID
id2grained = {v: k for k, v in grained2id.items()}

id_relation = {}  # {coarse_id:[grained_ids in coarse],...}
for coarse, grained_list in constant.GRAINED_ID_IN_COARSE.items():
    grained_list_tmp = list(grained_list.keys())
    grained_list = []
    for grained in grained_list_tmp:
        if grained in grained2id:
            grained_list.append(grained2id[grained])
    id_relation[coarse2id[coarse]] = grained_list

# testset labels predictions
test_set = pd.read_csv("../dataset/test.tsv", "\t", header=None)  # label text
labels = eval(open('grained/labels').read())
grained_preds = eval(open('grained/preds').read())
grained_preds_top5 = eval(open('grained/preds_top5').read())  # [[label id,logistics],...]
coarse_preds_top5 = eval(open('coarse/preds_top5').read())
num = len(labels)

preds = []
for k in range(num):
    tmp = {}
    index, product = 0, 0
    coarse_preds_top5[k] = softmax(coarse_preds_top5[k])  # [[label id,softmax],...]
    grained_preds_top5[k] = softmax(grained_preds_top5[k])

    for i, coarse_item in enumerate(coarse_preds_top5[k]):
        for j, grained_item in enumerate(grained_preds_top5[k]):
            if grained_item[0] not in id_relation[coarse_item[0]]:
                product = 0
            else:
                product = grained_item[1] * coarse_item[1]
            tmp[index] = product
            index += 1
    tmp = list(sorted(tmp.items(), key=lambda x: x[1], reverse=True))
    max_index = list(tmp[0])[0]
    preds.append(grained_preds_top5[k][max_index % 5][0])

with open("grained/preds_top5_probs", 'w') as f:
    f.write(str(grained_preds_top5))
with open("coarse/preds_top5_probs", 'w') as f:
    f.write(str(coarse_preds_top5))

acc = metrics.accuracy_score(labels, preds)
f1 = metrics.f1_score(labels,preds,average='macro')
grained_acc = metrics.accuracy_score(labels, grained_preds)
grained_f1 = metrics.f1_score(labels, grained_preds,average='macro')
print("acc:{} f1:{}".format(acc,f1))
print("grained acc:{} f1:{}".format(grained_acc,grained_f1))

#
# wrong_preds = []
# for i, item in enumerate(preds):
#     if preds[i] != labels[i]:
#         wrong_item = [test_set[1][i], id2grained[labels[i]], id2grained[preds[i]]]
#         wrong_preds.append(wrong_item)
#
#
# # wrong_preds.sort(key=lambda x:x[1])
# #
# # df = pd.DataFrame(wrong_preds)
# # df.columns = ['text', 'correct_label', 'wrong_prediction']
# # df.to_excel('./wrong_preds.xlsx', index=False)
# # print("wrong prediction saved to result/wrong_preds.xlsx")