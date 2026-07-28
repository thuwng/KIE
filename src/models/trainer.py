# src/models/trainer.py
import torch
from transformers import Trainer

class CompletionOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        completion_lens = (labels != -100).sum(dim=1)
        keep = completion_lens.max().item() + 1  # +1: cần token cuối của prompt để dự đoán token đầu completion

        outputs = model(**inputs, num_logits_to_keep=keep)
        logits = outputs.logits  # (batch, keep, vocab) — chỉ nhỏ bằng completion, không phải cả document

        labels_window = labels[:, -keep:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels_window[:, 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss