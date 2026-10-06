#!/usr/bin/env python
# coding=utf-8
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset
import evaluate
import transformers
import torch
from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

check_min_version("4.5.0")

logger = logging.getLogger(__name__)
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(default="ner")
    dataset_name: Optional[str] = field(default='funsd', metadata={"help": "funsd | cord"})
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "funsd | funsd_word | funsd_word_ro | cord | cord_word (mặc định = config gốc)"},
    )
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_val_samples: Optional[int] = field(default=None)
    max_test_samples: Optional[int] = field(default=None)
    label_all_tokens: bool = field(default=False)
    return_entity_level_metrics: bool = field(default=False)
    # NEW: tập dev để chọn siêu tham số. 0 = không tách dev (dùng cho lượt chạy cuối, số bước cố định).
    dev_ratio: float = field(
        default=0.0,
        metadata={"help": "Tỉ lệ train tách làm dev (FUNSD không có dev). Dataset có sẵn validation thì dùng luôn."},
    )
    dev_split_seed: int = field(default=42, metadata={"help": "Seed CỐ ĐỊNH cho việc tách dev, độc lập seed train."})
    visual_embed: bool = field(default=True)
    # ---- NEW: Latent Soft Segment ----
    use_latent_segment: bool = field(default=False, metadata={"help": "Dùng LayoutLMv3ForLatentSegmentTokenClassification"})
    latent_layer: int = field(default=6, metadata={"help": "Số lớp chạy ở lượt 1 để tính affinity"})
    latent_soft_box: bool = field(default=True, metadata={"help": "Ablation: tắt box mềm"})
    latent_attn_bias: bool = field(default=True, metadata={"help": "Ablation: tắt bias attention"})
    latent_tf_ratio: float = field(default=0.5, metadata={"help": "Teacher forcing giảm 1->0 trong tỉ lệ này của tổng bước; 0 = tắt"})
    latent_oracle_eval: bool = field(default=False, metadata={"help": "CẬN TRÊN: dùng nhóm gold lúc test (không phải kết quả chính)"})
    affinity_loss_weight: float = field(default=1.0)
    head_learning_rate: float = field(default=5e-4, metadata={"help": "LR cho các tham số latent_* mới"})
    input_size: int = field(default=224)
    second_input_size: int = field(default=112)
    train_interpolation: str = field(default='bicubic')
    second_interpolation: str = field(default='lanczos')
    imagenet_default_mean_and_std: bool = field(default=False)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # ------------------------------------------------------------ dataset
    if data_args.dataset_name == 'funsd':
        import layoutlmft.data.funsd as ds_module
    elif data_args.dataset_name == 'cord':
        import layoutlmft.data.cord as ds_module
    else:
        raise NotImplementedError()
    datasets = load_dataset(os.path.abspath(ds_module.__file__), name=data_args.dataset_config_name,
                            cache_dir=model_args.cache_dir)

    # NEW: tách dev cố định (không bao giờ chọn mô hình trên test)
    dev_raw = None
    if training_args.do_eval:
        if "validation" in datasets:
            dev_raw = datasets["validation"]
        elif data_args.dev_ratio > 0:
            split = datasets["train"].train_test_split(test_size=data_args.dev_ratio, seed=data_args.dev_split_seed)
            datasets["train"], dev_raw = split["train"], split["test"]
        else:
            raise ValueError("--do_eval cần tập dev: đặt --dev_ratio > 0 (KHÔNG đánh giá chọn mô hình trên test).")

    split_for_meta = "train" if training_args.do_train else "test"
    column_names = datasets[split_for_meta].column_names
    features = datasets[split_for_meta].features
    text_column_name = "words" if "words" in column_names else "tokens"
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )
    remove_columns = column_names

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        unique = set()
        for l in datasets["train"][label_column_name]:
            unique |= set(l)
        label_list = sorted(unique)
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # ------------------------------------------------------------ model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        visual_embed=data_args.visual_embed,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_kwargs = dict(
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.use_latent_segment:
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import (
            LayoutLMv3ForLatentSegmentTokenClassification,
        )
        config.latent_layer = data_args.latent_layer
        config.latent_soft_box = data_args.latent_soft_box
        config.latent_attn_bias = data_args.latent_attn_bias
        config.latent_oracle_eval = data_args.latent_oracle_eval
        config.affinity_loss_weight = data_args.affinity_loss_weight
        model = LayoutLMv3ForLatentSegmentTokenClassification.from_pretrained(
            model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("This example script only works for models that have a fast tokenizer.")

    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.visual_embed:
        mean = IMAGENET_INCEPTION_MEAN if not data_args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not data_args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            RandomResizedCropAndInterpolationWithTwoPic(
                size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])
        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ])

    # ------------------------------------------------------------ tokenize
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,  # tài liệu > 512 token bị cắt thành nhiều đoạn KHÔNG chồng lấn
            is_split_into_words=True,
        )
        labels, bboxes, images, group_ids, doc_ids, word_idxs = [], [], [], [], [], []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org = tokenized_inputs["overflow_to_sample_mapping"][batch_index]
            label = examples[label_column_name][org]
            bbox = examples["bboxes"][org]

            # id nhóm gold cho từng từ (chỉ dùng làm nhãn lúc train). Nếu dataset chưa có cột
            # group_ids (vd. cord.py chưa sửa) thì suy ra từ box bằng nhau như code cũ.
            if "group_ids" in examples:
                word_gid = examples["group_ids"][org]
            else:
                word_gid, c, prev = [], -1, None
                for wb in bbox:
                    if tuple(wb) != prev:
                        c, prev = c + 1, tuple(wb)
                    word_gid.append(c)

            previous_word_idx = None
            label_ids, bbox_inputs, gid_inputs, widx_inputs = [], [], [], []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                    gid_inputs.append(-1)
                    widx_inputs.append(-1)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                    gid_inputs.append(word_gid[word_idx])
                    widx_inputs.append(word_idx)          # sub-token ĐẦU của từ -> dùng để chấm điểm
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                    gid_inputs.append(word_gid[word_idx])
                    widx_inputs.append(-1)
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            group_ids.append(gid_inputs)
            doc_ids.append(examples["id"][org])
            word_idxs.append(widx_inputs)

            if data_args.visual_embed:
                img = pil_loader(examples["image_path"][org])
                for_patches, _ = common_transform(img, augmentation=augmentation)
                images.append(patch_transform(for_patches))

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["group_ids"] = group_ids   # Trainer tự bỏ cột này nếu forward() không nhận
        tokenized_inputs["doc_id"] = doc_ids        # chỉ dùng để chấm điểm cấp tài liệu (Trainer tự bỏ)
        tokenized_inputs["word_idx"] = word_idxs    # chỉ dùng để chấm điểm (Trainer tự bỏ)
        if data_args.visual_embed:
            tokenized_inputs["images"] = images
        return tokenized_inputs

    def prep(ds, max_n):
        if max_n is not None:
            ds = ds.select(range(max_n))
        return ds.map(tokenize_and_align_labels, batched=True, remove_columns=remove_columns,
                      num_proc=data_args.preprocessing_num_workers,
                      load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = prep(datasets["train"], data_args.max_train_samples) if training_args.do_train else None
    eval_dataset = prep(dev_raw, data_args.max_val_samples) if training_args.do_eval else None
    test_dataset = prep(datasets["test"], data_args.max_test_samples) if training_args.do_predict else None

    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # ------------------------------------------------------------ METRIC (chốt 1 cách)
    # seqeval entity-level micro P/R/F1, sơ đồ IOB2, chế độ mặc định (giống LayoutLMv3 gốc),
    # chấm trên sub-token ĐẦU của mỗi từ, GỘP các đoạn 512 về NGUYÊN TÀI LIỆU, mỗi từ đúng 1 lần.
    metric = evaluate.load("seqeval")

    def doc_level_sequences(predictions, label_ids, ds):
        predictions = np.argmax(predictions, axis=2)
        doc_ids, word_idx = ds["doc_id"], ds["word_idx"]
        docs, order = {}, []
        for i in range(len(predictions)):
            d = doc_ids[i]
            if d not in docs:
                docs[d] = {}
                order.append(d)
            seen = docs[d]
            for j, w in enumerate(word_idx[i]):
                if w < 0 or label_ids[i][j] == -100 or w in seen:
                    continue
                seen[w] = (label_list[predictions[i][j]], label_list[label_ids[i][j]])
        y_pred, y_true = [], []
        for d in order:
            items = sorted(docs[d].items())
            y_pred.append([p for _, (p, _) in items])
            y_true.append([g for _, (_, g) in items])
        return order, y_pred, y_true

    def make_compute_metrics(ds):
        def compute_metrics(p):
            _, y_pred, y_true = doc_level_sequences(p.predictions, p.label_ids, ds)
            results = metric.compute(predictions=y_pred, references=y_true)
            out = {"precision": results["overall_precision"], "recall": results["overall_recall"],
                   "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
            if data_args.return_entity_level_metrics:
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            out[f"{key}_{n}"] = v
            return out
        return compute_metrics

    # ------------------------------------------------------------ trainer
    class LatentTrainer(Trainer):
        """LR riêng cho module latent_*; cập nhật xác suất teacher forcing theo bước."""

        def create_optimizer(self):
            if self.optimizer is None:
                base, head = [], []
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        (head if n.startswith("latent_") else base).append(p)
                self.optimizer = torch.optim.AdamW(
                    [{"params": base, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
                     {"params": head, "lr": data_args.head_learning_rate, "weight_decay": 0.0}],
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            return self.optimizer

        def training_step(self, model, inputs, *args, **kwargs):
            r = data_args.latent_tf_ratio
            total = max(1, self.state.max_steps)
            tf = max(0.0, 1.0 - self.state.global_step / (r * total)) if r > 0 else 0.0
            self.model.latent_tf_prob = tf
            return super().training_step(model, inputs, *args, **kwargs)

    TrainerCls = LatentTrainer if data_args.use_latent_segment else Trainer  # baseline = Trainer gốc
    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(eval_dataset) if eval_dataset is not None else None,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate on DEV ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict on TEST (checkpoint cuối, số bước cố định) ***")
        trainer.compute_metrics = make_compute_metrics(test_dataset)
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        order, y_pred, y_true = doc_level_sequences(predictions, label_ids, test_dataset)
        if trainer.is_world_process_zero():
            with open(os.path.join(training_args.output_dir, "test_predictions.jsonl"), "w") as w:
                for d, yp, yt in zip(order, y_pred, y_true):
                    w.write(json.dumps({"doc_id": d, "pred": yp, "gold": yt}) + "\n")


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()