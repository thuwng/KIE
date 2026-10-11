# coding=utf-8
"""
Dựng chuỗi đầu vào (thứ tự từ, box 2D, nhãn IOB) cho FUNSD / CORD theo từng setting.

Hai trục độc lập:
  box_level : "segment" = box của KHỐI ANNOTATION GOLD (đúng code gốc LayoutLMv3: get_line_bbox trên mỗi item)
              "word"    = box riêng của từng từ
              "line"    = box của "dòng" gom bằng hình học thuần (cùng hàng + khoảng cách ngang nhỏ),
                          KHÔNG nhìn nhãn/khối gold -> giống segment do OCR trả về ngoài thực tế
  order     : "annotation" = thứ tự item trong file json (từ của một entity luôn liền nhau, đúng thứ tự)
              "reading"    = hàng trên->dưới, trong hàng trái->phải, chỉ dựa vào hình học

Nhãn IOB luôn được dựng lại từ (nhãn entity, id entity) theo thứ tự đầu vào cuối cùng:
  từ đầu tiên của một đoạn liên tiếp cùng entity -> B-, các từ sau -> I-, item "other" -> O.
Với order="annotation" kết quả TRÙNG KHỚP code gốc. Với order="reading", entity bị cắt rời
(vd: câu hỏi 2 dòng xen với câu trả lời bên phải) sẽ thành nhiều entity gold -> được đếm trong stats.

Mọi heuristic chạy trên box PIXEL (trước chuẩn hoá 0..1000) để tỉ lệ ngang/dọc không bị méo.
"""

BOX_LEVELS = ("segment", "word", "line")
ORDERS = ("annotation", "reading")


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def group_rows(boxes, min_overlap=0.5, lookback=8):
    """Gom từ thành hàng. Trả về list hàng (list chỉ số), hàng trên->dưới, từ trong hàng trái->phải."""
    idx = sorted(range(len(boxes)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2.0, boxes[i][0]))
    rows = []  # [y0, y1, members]  (y0/y1 = trung bình chạy của thành viên)
    for i in idx:
        _, y0, _, y1 = boxes[i]
        h = max(y1 - y0, 1)
        best, best_ov = None, 0.0
        for r in rows[-lookback:]:          # hàng được tạo theo tâm y tăng dần -> chỉ cần xét vài hàng gần nhất
            ov = _overlap(r[0], r[1], y0, y1) / min(h, max(r[1] - r[0], 1))
            if ov >= min_overlap and ov > best_ov:
                best, best_ov = r, ov
        if best is None:
            rows.append([float(y0), float(y1), [i]])
        else:
            best[2].append(i)
            m = len(best[2])
            best[0] += (y0 - best[0]) / m
            best[1] += (y1 - best[1]) / m
    rows.sort(key=lambda r: (r[0] + r[1]) / 2.0)
    return [sorted(r[2], key=lambda i: (boxes[i][0], boxes[i][1])) for r in rows]


def split_row(row, boxes, gap_ratio=1.0):
    """Cắt một hàng (đã sắp trái->phải) thành các dòng/segment khi khoảng trống ngang > gap_ratio * chiều cao chữ."""
    hs = sorted(max(boxes[i][3] - boxes[i][1], 1) for i in row)
    h = hs[len(hs) // 2]
    segs, cur = [], [row[0]]
    for a, b in zip(row, row[1:]):
        if boxes[b][0] - boxes[a][2] > gap_ratio * h:
            segs.append(cur)
            cur = []
        cur.append(b)
    segs.append(cur)
    return segs


def _union(bs):
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def retag_iob(seq, ent_labels, ent_ids):
    tags, prev = [], None
    for i in seq:
        lab = ent_labels[i]
        if lab == "O":
            tags.append("O")
        else:
            tags.append(("I-" if prev == ent_ids[i] else "B-") + lab)
        prev = ent_ids[i]
    return tags


def build_document(boxes, ent_labels, ent_ids, size, normalize_bbox, box_level, order, gap_ratio=1.0):
    """
    boxes      : box pixel [x0,y0,x1,y1] của từng từ, theo thứ tự annotation
    ent_labels : nhãn entity của từ ("O" hoặc "QUESTION"/"ANSWER"/...), ent_ids: id item chứa từ đó
    Trả về (seq, bboxes, tags, stats): seq = thứ tự chỉ số từ đưa vào model; bboxes/tags thẳng hàng với seq.
    """
    assert box_level in BOX_LEVELS and order in ORDERS, (box_level, order)
    n = len(boxes)
    norm = [normalize_bbox(b, size) for b in boxes]

    rows = group_rows(boxes) if (order == "reading" or box_level == "line") else None
    seq = list(range(n)) if order == "annotation" else [i for r in rows for i in r]

    if box_level == "word":
        box_of = norm
    else:
        groups = {}
        if box_level == "segment":                       # khối gold = các từ cùng item (gốc LayoutLMv3)
            for i in range(n):
                groups.setdefault(ent_ids[i], []).append(i)
            groups = list(groups.values())
        else:                                            # dòng hình học, không nhìn nhãn
            groups = [s for r in rows for s in split_row(r, boxes, gap_ratio)]
        box_of = [None] * n
        for g in groups:
            u = _union([norm[i] for i in g])
            for i in g:
                box_of[i] = u

    tags = retag_iob(seq, ent_labels, ent_ids)

    ents = {ent_ids[i] for i in range(n) if ent_labels[i] != "O"}
    pos = {}
    for k, i in enumerate(seq):
        pos.setdefault(ent_ids[i], []).append(k)
    n_split = sum(1 for e in ents if max(pos[e]) - min(pos[e]) + 1 != len(pos[e]))
    stats = {"entities": len(ents), "entities_after_retag": sum(t.startswith("B-") for t in tags),
             "entities_non_contiguous": n_split}
    return seq, [box_of[i] for i in seq], tags, stats