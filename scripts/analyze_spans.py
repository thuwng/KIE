#!/usr/bin/env python
"""
Phân tích ranh giới entity từ test_predictions.jsonl (không cần GPU, chạy vài giây).

    python scripts/analyze_spans.py logs/funsd-final-C-B0-seed42 logs/funsd-final-C-LSS-L8-seed42 ...

Với mỗi thư mục in ra:
  - pair P/R/F1: coi mỗi entity dự đoán là một "nhóm", so cặp từ cùng nhóm với entity gold.
    So được trực tiếp với test_group_pair_* của đầu gom nhóm LSS -> biết đầu BIO của chính
    mô hình gom nhóm tốt hơn hay kém hơn đầu affinity.
  - MERGE: 1 entity dự đoán phủ >= 2 entity gold (gộp quá tay)
  - SPLIT: 1 entity gold bị cắt thành >= 2 entity dự đoán (tách quá tay)
"""
import json
import os
import sys

from seqeval.metrics.sequence_labeling import get_entities


def groups(tags):
    g = [-1] * len(tags)
    for k, (_, s, e) in enumerate(get_entities(tags)):
        for i in range(s, e + 1):
            g[i] = k
    return g


def analyze(path):
    f = os.path.join(path, "test_predictions.jsonl") if os.path.isdir(path) else path
    tp = fp = fn = merge = split = n_pred = n_gold = 0
    for line in open(f, encoding="utf-8"):
        d = json.loads(line)
        gp, gg = groups(d["pred"]), groups(d["gold"])
        n = len(gp)
        for i in range(n):
            for j in range(i + 1, n):
                sp = gp[i] >= 0 and gp[i] == gp[j]
                sg = gg[i] >= 0 and gg[i] == gg[j]
                tp += sp and sg
                fp += sp and not sg
                fn += sg and not sp
        pe, ge = get_entities(d["pred"]), get_entities(d["gold"])
        n_pred += len(pe)
        n_gold += len(ge)
        for _, s, e in pe:
            if len({gg[i] for i in range(s, e + 1) if gg[i] >= 0}) >= 2:
                merge += 1
        for _, s, e in ge:
            if len({gp[i] for i in range(s, e + 1) if gp[i] >= 0}) >= 2:
                split += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(pair_P=p, pair_R=r, pair_F1=f1, MERGE=merge, SPLIT=split, n_pred=n_pred, n_gold=n_gold)


if __name__ == "__main__":
    print(f"{'run':45s} {'pairP':>6s} {'pairR':>6s} {'pairF1':>6s} {'MERGE':>6s} {'SPLIT':>6s} {'n_pred':>6s} {'n_gold':>6s}")
    for p in sys.argv[1:]:
        r = analyze(p)
        print(f"{os.path.basename(p.rstrip('/')):45s} {r['pair_P']:6.3f} {r['pair_R']:6.3f} {r['pair_F1']:6.3f} "
              f"{r['MERGE']:6d} {r['SPLIT']:6d} {r['n_pred']:6d} {r['n_gold']:6d}")