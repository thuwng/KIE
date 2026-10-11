# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py  (bản trong repo LayoutLMv3)

Config (tên config = cách dựng đầu vào; nhãn / ảnh / split giữ nguyên):
  funsd          A  box KHỐI GOLD + thứ tự annotation  = ĐÚNG code gốc LayoutLMv3 (90.29 base / 92.08 large)
  funsd_word     B  box TỪ        + thứ tự annotation  (bỏ rò rỉ qua box, thứ tự vẫn gom theo entity)
  funsd_word_ro  C  box TỪ        + thứ tự đọc         (không dùng gì từ khối gold: setting thực tế)
  funsd_line_ro  D  box DÒNG gom hình học + thứ tự đọc (segment-level như LayoutLMv3 nhưng segment từ hình học,
                                                        giống OCR line, không dùng khối gold)
'''
import json
import os
from collections import Counter

import datasets

from layoutlmft.data.image_utils import load_image, normalize_bbox
from layoutlmft.data.layout_settings import build_document


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

    def __init__(self, box_level="segment", order="annotation", **kwargs):
        super(FunsdConfig, self).__init__(**kwargs)
        self.box_level = box_level
        self.order = order


class Funsd(datasets.GeneratorBasedBuilder):
    """FUNSD dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"), box_level="segment", order="annotation",
                    description="A: box khối gold + thứ tự annotation (gốc LayoutLMv3)"),
        FunsdConfig(name="funsd_word", version=datasets.Version("1.0.0"), box_level="word", order="annotation",
                    description="B: box từ + thứ tự annotation"),
        FunsdConfig(name="funsd_word_ro", version=datasets.Version("1.0.0"), box_level="word", order="reading",
                    description="C: box từ + thứ tự đọc"),
        FunsdConfig(name="funsd_line_ro", version=datasets.Version("1.0.0"), box_level="line", order="reading",
                    description="D: box dòng (hình học) + thứ tự đọc"),
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

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s | config %s (box=%s, order=%s)",
                    filepath, self.config.name, self.config.box_level, self.config.order)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        total = Counter()
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)

            words, boxes, ent_labels, ent_ids = [], [], [], []
            for k, item in enumerate(data["form"]):
                ws = [w for w in item["words"] if w["text"].strip() != ""]
                lab = "O" if item["label"] == "other" else item["label"].upper()
                for w in ws:
                    words.append(w["text"])
                    boxes.append(w["box"])
                    ent_labels.append(lab)
                    ent_ids.append(k)

            seq, bboxes, tags, st = build_document(boxes, ent_labels, ent_ids, size, normalize_bbox,
                                                   self.config.box_level, self.config.order)
            total.update(st)
            yield guid, {"id": str(guid), "tokens": [words[i] for i in seq], "bboxes": bboxes, "ner_tags": tags,
                         "image": image, "image_path": image_path}
        logger.info("FUNSD %s | %s: %s", self.config.name, filepath, dict(total))