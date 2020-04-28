#!/usr/bin/env python
# usage: ./evaluation_absita2018.py RESULT_FILE GOLD_FILE
import sys
import csv

aspects = ['value', 'wifi', 'comfort', 'staff', 'location', 'amenities', 'cleanliness']
result = dict()
result_presence = dict()
result_polarity = dict()
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f, delimiter=';', quotechar='"')
    for row in reader:
        result_presence[row['sentence_id']] = set()
        result_polarity[row['sentence_id']] = set()
        result[row['sentence_id']] = {k:v for k, v in row.items()}
        for k, v in row.items():
            if v == '1':
                if k.endswith("_presence"):
                    result_presence[row['sentence_id']].add(k)
                elif k.endswith("_positive") or k.endswith("_negative"):
                    result_polarity[row['sentence_id']].add(k)


gold = dict()
gold_presence = dict()
gold_polarity = dict()
with open(sys.argv[2]) as f:
    reader = csv.DictReader(f, delimiter=';', quotechar='"')
    for row in reader:
        gold_presence[row['sentence_id']] = set()
        gold_polarity[row['sentence_id']] = set()
        gold[row['sentence_id']] = {k:v for k, v in row.items()}
        for k, v in row.items():
            if v == '1':
                if k.endswith("_presence"):
                    gold_presence[row['sentence_id']].add(k)
                elif k.endswith("_positive") or k.endswith("_negative"):
                    gold_polarity[row['sentence_id']].add(k)
# Evaluation ACD
precision = 0.0
recall = 0.0
hits = 0
predicted = 0
golden = 0
for sentence_id in gold_presence.keys():
    if len(result_presence[sentence_id]) > 0:
        precision += float(len(result_presence[sentence_id].intersection(gold_presence[sentence_id])))/float(len(result_presence[sentence_id]))
    if len(gold_presence[sentence_id]) > 0:
        recall += float(len(result_presence[sentence_id].intersection(gold_presence[sentence_id])))/float(len(gold_presence[sentence_id]))
    if precision + recall > 0.0:
        fscore = (2.0 * precision * recall) / (precision + recall)
    else:
        fscore = 0.0
    predicted += len(result_presence[sentence_id])
    golden += len(gold_presence[sentence_id])
    hits += len(result_presence[sentence_id].intersection(gold_presence[sentence_id]))
macro_precision = precision/float(len(gold_presence.keys()))
macro_recall = recall/float(len(gold_presence.keys()))
macro_fscore = fscore/float(len(gold_presence.keys()))

print("Task ACD:")
print("\tMacro-Precision: {:.4f}".format(macro_precision))
print("\tMacro-Recall: {:.4f}".format(macro_recall))
print("\tMacro-F1-score: {:.4f}".format(macro_fscore))
if predicted > 0:
    micro_precision = float(hits)/float(predicted)
else:
    micro_precision = 0.0
if golden > 0:
    micro_recall = float(hits)/float(golden)
else:
    micro_recall = 0.0
if micro_precision + micro_recall > 0:
    micro_fscore = (2.0 * micro_precision * micro_recall)/(micro_precision + micro_recall)
else:
    micro_fscore = 0.0  
print("\tMicro-Precision: {:.4f}".format(micro_precision))
print("\tMicro-Recall: {:.4f}".format(micro_recall))
print("\tMicro-F1-score: {:.4f}".format(micro_fscore))

# Evaluation ACP
precision = 0.0
recall = 0.0
hits = 0
predicted = 0
golden = 0
for sentence_id in gold_polarity.keys():
    if len(result_polarity[sentence_id]) > 0:
        precision += float(len(result_polarity[sentence_id].intersection(gold_polarity[sentence_id])))/float(len(result_polarity[sentence_id]))
    if len(gold_polarity[sentence_id]) > 0:
        recall += float(len(result_polarity[sentence_id].intersection(gold_polarity[sentence_id])))/float(len(gold_polarity[sentence_id]))
    if precision + recall > 0.0:
        fscore = (2.0 * precision * recall) / (precision + recall)
    else:
         fscore = 0.0
    predicted += len(result_polarity[sentence_id])
    golden += len(gold_polarity[sentence_id])
    hits += len(result_polarity[sentence_id].intersection(gold_polarity[sentence_id]))
macro_precision = precision/float(len(gold_presence.keys()))
macro_recall = recall/float(len(gold_presence.keys()))
macro_fscore = fscore/float(len(gold_presence.keys()))

print("Task ACP:")
print("\tMacro-Precision: {:.4f}".format(macro_precision))
print("\tMacro-Recall: {:.4f}".format(macro_recall))
print("\tMacro-F1-score: {:.4f}".format(macro_fscore))
if predicted > 0:
    micro_precision = float(hits)/float(predicted)
else:
    micro_precision = 0.0
if golden > 0:
    micro_recall = float(hits)/float(golden)
else:
    micro_recall = 0.0
if micro_precision + micro_recall > 0:
    micro_fscore = (2.0 * micro_precision * micro_recall)/(micro_precision + micro_recall)
else:
    micro_fscore = 0.0  
print("\tMicro-Precision: {:.4f}".format(micro_precision))
print("\tMicro-Recall: {:.4f}".format(micro_recall))
print("\tMicro-F1-score: {:.4f}".format(micro_fscore))
