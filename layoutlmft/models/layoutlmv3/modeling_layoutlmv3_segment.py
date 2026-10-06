# layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForLatentSegmentTokenClassification  ("Latent Soft Segment", LSS)

Vấn đề: LayoutLMv3 được pretrain với box cấp SEGMENT. Khi test thực tế chỉ có box
cấp TỪ (không có nhóm segment gold) -> lệch phân phối so với pretrain -> tụt 8-10 F1.

Ý tưởng: cho mô hình TỰ suy ra nhóm segment (mềm, khả vi) ngay khi fine-tune,
rồi đưa nhóm này ngược lại vào (1) embedding 2D và (2) attention.

  Lượt 1 (dùng chung trọng số, chỉ chạy K lớp đầu): box từ + ảnh -> hidden h^(K)
  Đầu affinity:   A_ij = sigmoid( <q_i, k_j>/sqrt(d) + MLP(hình học_ij) )   (đối xứng)
  Box mềm:        soft-union có trọng số A (soft-min x0,y0 ; soft-max x1,y1)
  Lượt 2 (full):  E2D = a*E(box từ) + (1-a)*E_interp(box mềm)     [khả vi theo A]
                  attention += lambda_h * log A_ij  (text-text, lambda theo từng head)
  Loss = CE(token) + w * BCE cân bằng(A, nhóm gold)      [nhóm gold CHỈ dùng khi train]
  Scheduled sampling: đầu train thay A bằng nhóm gold với xác suất p giảm tuyến tính về 0.

KHÔNG dùng group_ids lúc eval/test (trừ khi bật latent_oracle_eval để đo cận trên).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)


class LayoutLMv3ForLatentSegmentTokenClassification(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"latent_"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        H = config.hidden_size

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Đầu phân loại GIỮ NGUYÊN như baseline (công bằng khi so sánh).
        if config.num_labels < 10:
            self.classifier = nn.Linear(H, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # ---- cấu hình (đọc từ config, có giá trị mặc định an toàn) ----
        self.latent_layer = getattr(config, "latent_layer", 6)
        self.use_soft_box = getattr(config, "latent_soft_box", True)
        self.use_attn_bias = getattr(config, "latent_attn_bias", True)
        self.oracle_eval = getattr(config, "latent_oracle_eval", False)
        self.aff_weight = getattr(config, "affinity_loss_weight", 1.0)
        self.union_tau = getattr(config, "latent_union_tau", 3.0)       # độ "mềm" của soft-min/max (đơn vị toạ độ 0..1000)
        self.union_penalty = getattr(config, "latent_union_penalty", 1000.0)  # phạt (1-A)*M để loại từ khác nhóm
        d = getattr(config, "latent_affinity_dim", 128)

        # ---- module mới: TÊN BẮT ĐẦU BẰNG "latent_" để optimizer cho LR riêng ----
        self.latent_q = nn.Linear(H, d)
        self.latent_k = nn.Linear(H, d)
        self.latent_geo = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1))
        # a = sigmoid(latent_alpha) là trọng số của box TỪ; khởi tạo -2 -> a~0.12,
        # tức ưu tiên box mềm (gần prior segment lúc pretrain).
        self.latent_alpha = nn.Parameter(torch.full((1,), -2.0))
        # lambda theo từng head, khởi tạo 0 -> lúc đầu attention y hệt baseline.
        self.latent_attn_lambda = nn.Parameter(torch.zeros(config.num_attention_heads))

        # Xác suất teacher forcing, Trainer cập nhật mỗi bước (không phải tham số).
        self.latent_tf_prob = 0.0
        # Bộ đếm để LOG (không phải tham số, không ảnh hưởng tính toán): ce/aff loss, tỉ lệ teacher forcing,
        # và chất lượng gom nhóm pairwise (A>0.5 so với nhóm gold) lúc eval/test.
        self._stats = {}

        self.init_weights()

    # ------------------------------------------------------------------ log helpers
    def _acc(self, key, value, n=1.0):
        s = self._stats.setdefault(key, [0.0, 0.0])
        s[0] += float(value)
        s[1] += n

    def pop_latent_stats(self, prefix=""):
        """Trả về trung bình các đại lượng đã cộng dồn kể từ lần pop trước, rồi xoá bộ đếm."""
        st, self._stats = self._stats, {}
        out = {prefix + k: s / n for k, (s, n) in st.items() if not k.startswith("pair_") and n > 0}
        tp, fp, fn = (st.get(k, (0.0, 0.0))[0] for k in ("pair_tp", "pair_fp", "pair_fn"))
        if tp + fp + fn > 0:
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            out[prefix + "group_pair_precision"] = p
            out[prefix + "group_pair_recall"] = r
            out[prefix + "group_pair_f1"] = 2 * p * r / (p + r) if p + r else 0.0
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pair_geometry(box):
        """box: (B,T,4) float trong [0,1000] -> (B,T,T,8) đặc trưng hình học cặp (i,j)."""
        x0, y0, x1, y1 = box.unbind(-1)
        yc = (y0 + y1) / 2
        h = (y1 - y0).clamp(min=1.0)

        def d(a, b):  # a_j - b_i
            return a[:, None, :] - b[:, :, None]

        hgap_r = d(x0, x1)            # từ j nằm bên phải i bao xa
        hgap_l = -d(x1, x0)           # từ j nằm bên trái i bao xa
        vgap_d = d(y0, y1)            # j nằm dưới i
        vgap_u = -d(y1, y0)           # j nằm trên i
        dyc = d(yc, yc)
        dx0 = d(x0, x0)
        h_ratio = torch.log(h[:, None, :] / h[:, :, None])
        v_overlap = (torch.minimum(y1[:, None, :], y1[:, :, None]) - torch.maximum(y0[:, None, :], y0[:, :, None])) \
            / torch.minimum(h[:, None, :], h[:, :, None])
        feats = torch.stack([hgap_r, hgap_l, vgap_d, vgap_u, dyc, dx0], dim=-1) / 100.0
        feats = torch.cat([feats, h_ratio.unsqueeze(-1), v_overlap.unsqueeze(-1)], dim=-1)
        return feats.clamp(-10.0, 10.0)

    def _soft_union(self, box, A, valid):
        """
        Soft-union có trọng số A:  x0~_i = softmin_j [x0_j + (1-A_ij)*M]  (tương tự y0; x1,y1 dùng softmax).
        A_ii = 1 nên từ luôn thuộc nhóm của chính nó. Từ có A nhỏ bị phạt M -> gần như bị loại.
        Tính bằng float32 cho ổn định khi dùng fp16.
        """
        tau, M = self.union_tau, self.union_penalty
        box = box.float()
        pen = (1.0 - A.float()) * M                     # (B,T,T)
        col_ok = valid[:, None, :]                      # chỉ lấy j hợp lệ
        FILL = -1e4                                     # số hữu hạn, tránh NaN khi cả hàng bị che

        def smin(v):
            z = (-(v[:, None, :] + pen) / tau).masked_fill(~col_ok, FILL)
            return -tau * torch.logsumexp(z, dim=-1)

        def smax(v):
            z = ((v[:, None, :] - pen) / tau).masked_fill(~col_ok, FILL)
            return tau * torch.logsumexp(z, dim=-1)

        sx0, sy0 = smin(box[..., 0]), smin(box[..., 1])
        sx1, sy1 = smax(box[..., 2]), smax(box[..., 3])
        soft = torch.stack([sx0, sy0, sx1, sy1], dim=-1).clamp(0.0, 1000.0)
        soft = torch.stack([soft[..., 0], soft[..., 1],
                            torch.maximum(soft[..., 2], soft[..., 0]),
                            torch.maximum(soft[..., 3], soft[..., 1])], dim=-1)
        return torch.where(valid[..., None], soft, box)

    @staticmethod
    def _interp_embed(emb, v):
        """Tra embedding tại toạ độ THỰC bằng nội suy tuyến tính -> khả vi theo v."""
        n = emb.num_embeddings
        v = v.clamp(0.0, n - 1.001)
        lo = v.floor().long()
        hi = (lo + 1).clamp(max=n - 1)
        w = (v - lo.float()).unsqueeze(-1)
        return emb(lo) * (1.0 - w) + emb(hi) * w

    def _soft_spatial_embeddings(self, box):
        """Giống LayoutLMv3Embeddings._calc_spatial_position_embeddings nhưng nhận toạ độ thực."""
        E = self.layoutlmv3.embeddings
        x0, y0, x1, y1 = box.unbind(-1)
        return torch.cat([
            self._interp_embed(E.x_position_embeddings, x0),
            self._interp_embed(E.y_position_embeddings, y0),
            self._interp_embed(E.x_position_embeddings, x1),
            self._interp_embed(E.y_position_embeddings, y1),
            self._interp_embed(E.h_position_embeddings, (y1 - y0).clamp(0.0, 1023.0)),
            self._interp_embed(E.w_position_embeddings, (x1 - x0).clamp(0.0, 1023.0)),
        ], dim=-1)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        group_ids=None,  # (B,T) id nhóm gold, -1 cho special/pad. CHỈ dùng khi train (hoặc oracle).
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B, T = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)

        # Token "từ thật": không phải <s>, </s>, <pad> (không dùng group_ids -> không rò rỉ nhãn).
        cfg = self.config
        special = (input_ids == cfg.pad_token_id) | (input_ids == getattr(cfg, "bos_token_id", 0)) \
            | (input_ids == getattr(cfg, "eos_token_id", 2))
        valid = (attention_mask[:, :T] == 1) & ~special                       # (B,T)
        pair_mask = valid[:, :, None] & valid[:, None, :]                      # (B,T,T)
        eye = torch.eye(T, dtype=torch.bool, device=device)[None]

        # ---------------- Lượt 1: K lớp đầu, box từ + ảnh ----------------
        out1 = self.layoutlmv3(
            input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, return_dict=True, images=images,
            valid_span=valid_span, max_layers=self.latent_layer,
        )
        h = out1[0][:, :T, :]

        # ---------------- Affinity ----------------
        q, k = self.latent_q(h), self.latent_k(h)
        aff_logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        aff_logits = aff_logits.float() + self.latent_geo(self._pair_geometry(bbox.float())).squeeze(-1).float()
        aff_logits = 0.5 * (aff_logits + aff_logits.transpose(1, 2))          # đối xứng
        A = torch.sigmoid(aff_logits)
        A = torch.where(eye, torch.ones_like(A), A) * pair_mask.float()

        # Nhóm gold (chỉ cho loss / teacher forcing / oracle)
        G = None
        if group_ids is not None:
            G = (group_ids[:, :, None] == group_ids[:, None, :]) & (group_ids[:, :, None] >= 0) & pair_mask

        aff_loss = None
        if self.training and G is not None and self.aff_weight > 0:
            off = pair_mask & ~eye
            pos, neg = off & G, off & ~G
            bce = F.binary_cross_entropy_with_logits(aff_logits, G.float(), reduction="none")
            l_pos = (bce * pos).sum() / pos.sum().clamp(min=1)
            l_neg = (bce * neg).sum() / neg.sum().clamp(min=1)
            aff_loss = 0.5 * (l_pos + l_neg)                                  # BCE cân bằng dương/âm

        # LOG chất lượng gom nhóm lúc eval/test: cặp TỪ (sub-token đầu) trong cùng đoạn 512, ngưỡng 0.5.
        # Chỉ ĐO, không đưa vào dự đoán -> không rò rỉ nhãn.
        if (not self.training) and G is not None:
            with torch.no_grad():
                wmask = valid if labels is None else valid & (labels[:, :T] != -100)
                pm = wmask[:, :, None] & wmask[:, None, :] & ~eye
                pa = A > 0.5
                self._acc("pair_tp", (pa & G & pm).sum().item(), 0.0)
                self._acc("pair_fp", (pa & ~G & pm).sum().item(), 0.0)
                self._acc("pair_fn", (~pa & G & pm).sum().item(), 0.0)

        # Scheduled sampling (train) hoặc oracle (eval, chỉ để đo cận trên)
        A_used = A
        if G is not None:
            use_gold = None
            if self.training and self.latent_tf_prob > 0:
                use_gold = torch.rand(B, device=device) < self.latent_tf_prob
                self._acc("tf_used_frac", use_gold.float().mean().item())
            elif (not self.training) and self.oracle_eval:
                use_gold = torch.ones(B, dtype=torch.bool, device=device)
            if use_gold is not None:
                Gf = torch.where(eye, torch.ones_like(A), G.float()) * pair_mask.float()
                A_used = torch.where(use_gold[:, None, None], Gf, A)

        # ---------------- Box mềm -> embedding 2D ----------------
        spatial = None
        bbox2 = bbox
        if self.use_soft_box:
            soft_box = self._soft_union(bbox, A_used, valid)                    # (B,T,4) float
            E = self.layoutlmv3.embeddings
            spatial_word = E._calc_spatial_position_embeddings(bbox)
            a = torch.sigmoid(self.latent_alpha)
            spatial = a * spatial_word + (1.0 - a) * self._soft_spatial_embeddings(soft_box)
            # bias 2D tương đối của LayoutLMv3 cần toạ độ nguyên -> dùng box mềm làm tròn (không cần gradient)
            bbox2 = torch.where(valid[..., None], soft_box.detach().round().long(), bbox)

        # ---------------- Bias attention: lambda_h * log A ----------------
        extra_bias = None
        if self.use_attn_bias:
            L_tot = attention_mask.shape[1]                                    # text + ảnh
            logA = torch.log(A_used.clamp(min=1e-4))
            logA = torch.where(pair_mask, logA, torch.zeros_like(logA))
            extra_bias = self.latent_attn_lambda.view(1, -1, 1, 1) * logA.unsqueeze(1)   # (B,heads,T,T)
            if L_tot > T:
                extra_bias = F.pad(extra_bias, (0, L_tot - T, 0, L_tot - T))

        # ---------------- Lượt 2: full model ----------------
        outputs = self.layoutlmv3(
            input_ids, bbox=bbox2, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, images=images, valid_span=valid_span,
            spatial_position_embeddings=spatial, extra_attention_bias=extra_bias,
        )
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(logits.view(-1, self.num_labels), active_labels)
            if self.training:
                self._acc("ce_loss", loss.item())
            if aff_loss is not None:
                self._acc("aff_loss", aff_loss.item())
                loss = loss + self.aff_weight * aff_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )