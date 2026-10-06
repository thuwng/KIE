# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py

3 setting (chọn bằng --dataset_config_name):
  funsd          : box SEGMENT gold, thứ tự annotation  (giống gốc, CHỈ để tham chiếu - có rò rỉ nhãn)
  funsd_word     : box TỪ, thứ tự annotation            (so được với dòng "box từ ~81")
  funsd_word_ro  : box TỪ, sắp theo thứ tự đọc (trên->dưới, trái->phải)  <- SETTING CHÍNH (thực tế)
Cột "group_ids": id của item FUNSD (= segment gold) cho từng từ. Chỉ được dùng làm nhãn lúc train.
'''
import json
import os

import datasets

from layoutlmft.data.image_utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, segment_level_layout=True, reading_order=False, **kwargs):
        super(FunsdConfig, self).__init__(**kwargs)
        self.segment_level_layout = segment_level_layout
        self.reading_order = reading_order


def _reading_order(recs):
    """recs: list (text, box, gid, label). Gom thành dòng theo tâm y, trong dòng sắp theo x0.
    Mô phỏng thứ tự của OCR thông thường; KHÔNG dùng thông tin nhóm gold."""
    if not recs:
        return recs
    hs = sorted(max(1, r[1][3] - r[1][1]) for r in recs)
    tol = 0.5 * hs[len(hs) // 2]
    yc = lambda r: (r[1][1] + r[1][3]) / 2.0
    by_y = sorted(recs, key=lambda r: (yc(r), r[1][0]))
    lines, cur, cur_y = [], [], None
    for r in by_y:
        if cur and abs(yc(r) - cur_y) > tol:
            lines.append(cur)
            cur = []
        cur.append(r)
        cur_y = sum(yc(x) for x in cur) / len(cur)
    if cur:
        lines.append(cur)
    out = []
    for line in lines:
        out.extend(sorted(line, key=lambda r: r[1][0]))
    return out


class Funsd(datasets.GeneratorBasedBuilder):
    """FUNSD dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"),
                    description="gold segment boxes, annotation order",
                    segment_level_layout=True, reading_order=False),
        FunsdConfig(name="funsd_word", version=datasets.Version("1.0.0"),
                    description="word boxes, annotation order",
                    segment_level_layout=False, reading_order=False),
        FunsdConfig(name="funsd_word_ro", version=datasets.Version("1.0.0"),
                    description="word boxes, heuristic reading order",
                    segment_level_layout=False, reading_order=True),
    ]
    DEFAULT_CONFIG_NAME = "funsd"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "group_ids": datasets.Sequence(datasets.Value("int64")),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)

            recs = []  # (text, box, group_id, label)
            gid = 0
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                boxes = [normalize_bbox(w["box"], size) for w in words]
                if self.config.segment_level_layout:
                    boxes = self.get_line_bbox(boxes)
                for w, b in zip(words, boxes):
                    recs.append((w["text"], list(b), gid, label))
                gid += 1

            if self.config.reading_order:
                recs = _reading_order(recs)

            # Gán lại BIO theo thứ tự đầu vào: entity = đoạn liên tiếp các từ cùng nhóm.
            # Ở thứ tự annotation điều này trùng hệt code gốc (từ đầu item = B, còn lại = I).
            tokens, bboxes, ner_tags, group_ids = [], [], [], []
            prev_gid = None
            for text, b, g, label in recs:
                tokens.append(text)
                bboxes.append(b)
                group_ids.append(g)
                if label == "other":
                    ner_tags.append("O")
                else:
                    ner_tags.append(("I-" if g == prev_gid else "B-") + label.upper())
                prev_gid = g

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                         "group_ids": group_ids, "image": image, "image_path": image_path}