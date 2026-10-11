'''
Reference: https://huggingface.co/datasets/pierresi/cord/blob/main/cord.py  (bản trong repo LayoutLMv3)

Config giống FUNSD:  cord (A, gốc) | cord_word (B) | cord_word_ro (C) | cord_line_ro (D)
Khối gold của CORD = một "valid_line" (một dòng đã gán category).
'''
import json
import os
from collections import Counter
from pathlib import Path

import datasets

from layoutlmft.data.image_utils import load_image, normalize_bbox
from layoutlmft.data.layout_settings import build_document

logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
"""
_DESCRIPTION = """\
https://github.com/clovaai/cord/
"""

_CATS = ["MENU.NM", "MENU.NUM", "MENU.UNITPRICE", "MENU.CNT", "MENU.DISCOUNTPRICE", "MENU.PRICE", "MENU.ITEMSUBTOTAL",
         "MENU.VATYN", "MENU.ETC", "MENU.SUB_NM", "MENU.SUB_UNITPRICE", "MENU.SUB_CNT", "MENU.SUB_PRICE",
         "MENU.SUB_ETC", "VOID_MENU.NM", "VOID_MENU.PRICE", "SUB_TOTAL.SUBTOTAL_PRICE", "SUB_TOTAL.DISCOUNT_PRICE",
         "SUB_TOTAL.SERVICE_PRICE", "SUB_TOTAL.OTHERSVC_PRICE", "SUB_TOTAL.TAX_PRICE", "SUB_TOTAL.ETC",
         "TOTAL.TOTAL_PRICE", "TOTAL.TOTAL_ETC", "TOTAL.CASHPRICE", "TOTAL.CHANGEPRICE", "TOTAL.CREDITCARDPRICE",
         "TOTAL.EMONEYPRICE", "TOTAL.MENUTYPE_CNT", "TOTAL.MENUQTY_CNT"]
_LABELS = ["O"] + ["B-" + c for c in _CATS] + ["I-" + c for c in _CATS]   # đúng thứ tự gốc


def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        bbox[1], bbox[3] = bbox[3], bbox[1]
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        bbox[0], bbox[2] = bbox[2], bbox[0]
        box = tuple(bbox)
    return box


def _get_drive_url(url):
    base_url = 'https://drive.google.com/uc?id='
    split_url = url.split('/')
    return base_url + split_url[5]


_URLS = [
    _get_drive_url("https://drive.google.com/file/d/1MqhTbcj-AHXOqYoeoh12aRUwIprzTJYI/"),
    _get_drive_url("https://drive.google.com/file/d/1wYdp5nC9LnHQZ2FcmOoC0eClyWvcuARU/")
    # Nếu tải tự động lỗi: "https://layoutlm.blob.core.windows.net/cord/CORD-1k-001.zip",
    #                      "https://layoutlm.blob.core.windows.net/cord/CORD-1k-002.zip"
]


class CordConfig(datasets.BuilderConfig):
    """BuilderConfig for CORD"""

    def __init__(self, box_level="segment", order="annotation", **kwargs):
        super(CordConfig, self).__init__(**kwargs)
        self.box_level = box_level
        self.order = order


class Cord(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CordConfig(name="cord", version=datasets.Version("1.0.0"), box_level="segment", order="annotation",
                   description="A: box khối gold + thứ tự annotation (gốc LayoutLMv3)"),
        CordConfig(name="cord_word", version=datasets.Version("1.0.0"), box_level="word", order="annotation",
                   description="B: box từ + thứ tự annotation"),
        CordConfig(name="cord_word_ro", version=datasets.Version("1.0.0"), box_level="word", order="reading",
                   description="C: box từ + thứ tự đọc"),
        CordConfig(name="cord_line_ro", version=datasets.Version("1.0.0"), box_level="line", order="reading",
                   description="D: box dòng (hình học) + thứ tự đọc"),
    ]
    DEFAULT_CONFIG_NAME = "cord"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=_LABELS)),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="https://github.com/clovaai/cord/",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URLS)
        # move files from the second URL together with files from the first one.
        dest = Path(downloaded_file[0]) / "CORD"
        for split in ["train", "dev", "test"]:
            for file_type in ["image", "json"]:
                if split == "test" and file_type == "json":
                    continue
                src = Path(downloaded_file[1]) / "CORD" / split / file_type
                if not src.exists():          # đã gộp ở lần sinh config trước
                    continue
                for f in src.iterdir():
                    os.rename(f, dest / split / file_type / f.name)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest / "train"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dest / "dev"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": dest / "test"}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s | config %s (box=%s, order=%s)",
                    filepath, self.config.name, self.config.box_level, self.config.order)
        ann_dir = os.path.join(filepath, "json")
        img_dir = os.path.join(filepath, "image")
        total = Counter()
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)

            words, boxes, ent_labels, ent_ids = [], [], [], []
            for k, item in enumerate(data["valid_line"]):
                ws = [w for w in item["words"] if w["text"].strip() != ""]
                lab = "O" if item["category"] == "other" else item["category"].upper()
                for w in ws:
                    words.append(w["text"])
                    boxes.append(list(quad_to_box(w["quad"])))
                    ent_labels.append(lab)
                    ent_ids.append(k)

            seq, bboxes, tags, st = build_document(boxes, ent_labels, ent_ids, size, normalize_bbox,
                                                   self.config.box_level, self.config.order)
            total.update(st)
            yield guid, {"id": str(guid), "words": [words[i] for i in seq], "bboxes": bboxes, "ner_tags": tags,
                         "image": image, "image_path": image_path}
        logger.info("CORD %s | %s: %s", self.config.name, filepath, dict(total))