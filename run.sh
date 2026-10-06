#!/usr/bin/env bash
# =============================================================================
# run_latent_segment.sh — đặt ở thư mục gốc repo (cùng cấp examples/ và layoutlmft/)
#
#   bash run_latent_segment.sh tune      # B1: chọn latent_layer trên DEV (10% train), KHÔNG đụng test
#   LAYER=6 bash run_latent_segment.sh final     # B2: bảng chính, 5 seed, số bước cố định, chấm trên test
#   LAYER=6 bash run_latent_segment.sh ablation  # B3: ablation + cận trên oracle (cần chạy final trước)
#   bash run_latent_segment.sh report    # tổng hợp: mean ± std, delta theo cặp seed, file summary
#   bash run_latent_segment.sh smoke     # chạy thử 20 bước để kiểm tra code trước khi chạy thật
#
# METRIC CHÍNH: test_f1 = seqeval entity-level (IOB2, mode mặc định như LayoutLMv3 gốc), cấp tài liệu.
# Ảnh visualize tự sinh cho seed VIS_SEED (mặc định seed đầu tiên) ở mọi thí nghiệm có --do_predict.
#
# Biến tùy chỉnh: GPU=0 SEEDS="42 43 44 45 46" MODEL=microsoft/layoutlmv3-base OUT=runs
#                 BS=2 ACC=8 LAYER=6 KEEP_CKPT=0 EVAL_FLAG=--evaluation_strategy
# =============================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")" && pwd)}
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

GPU=${GPU:-0}
SEEDS=${SEEDS:-"42 43 44 45 46"}
read -r -a SEED_ARR <<< "$SEEDS"
VIS_SEED=${VIS_SEED:-${SEED_ARR[0]}}
MODEL=${MODEL:-microsoft/layoutlmv3-base}
OUT=${OUT:-runs}
STEPS=${STEPS:-1000}      # giống LayoutLMv3 gốc trên FUNSD
LR=${LR:-1e-5}
BS=${BS:-2}               # BS*ACC = 16 = batch hiệu dụng của paper (8 GPU x 2)
ACC=${ACC:-8}
LAYER=${LAYER:-6}         # cập nhật sau bước tune
KEEP_CKPT=${KEEP_CKPT:-0} # 0 = xoá trọng số sau khi chạy xong (trừ C, D cần cho oracle/vẽ lại)
EVAL_FLAG=${EVAL_FLAG:---evaluation_strategy}   # transformers mới đổi tên thành --eval_strategy
MAIN=funsd_word_ro        # SETTING CHÍNH: box từ + thứ tự đọc heuristic, không nhóm gold lúc test
STAGE=${1:-final}

mkdir -p "$OUT"
STATUS="$OUT/run_status.tsv"
[ -f "$STATUS" ] || printf 'time\texperiment\tconfig\tseed\tstatus\tseconds\toutput_dir\n' > "$STATUS"

COMMON=(--model_name_or_path "$MODEL" --dataset_name funsd
        --max_steps "$STEPS" --learning_rate "$LR"
        --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$ACC"
        --per_device_eval_batch_size 4 --fp16
        --save_strategy no --logging_steps 20 --report_to none
        --overwrite_output_dir --dataloader_num_workers 4 --input_size 224)

LATENT=(--use_latent_segment True --latent_layer "$LAYER")

# run <tên_thí_nghiệm> <dataset_config> <seed> [tham số thêm...]
run () {
  local name=$1 cfg=$2 seed=$3; shift 3
  local out="$OUT/$name/seed$seed"
  if [ -f "$out/test_results.json" ] || [ -f "$out/eval_results.json" ]; then
    echo ">> bỏ qua (đã có kết quả): $out"; return 0; fi
  mkdir -p "$out"
  local vis=()
  [ "$seed" = "$VIS_SEED" ] && vis=(--visualize True)
  local cmd=(python examples/run_funsd_cord.py "${COMMON[@]}" --dataset_config_name "$cfg"
             --seed "$seed" --output_dir "$out" ${vis[@]+"${vis[@]}"} "$@")
  printf '%q ' "${cmd[@]}" > "$out/cmd.sh"; echo >> "$out/cmd.sh"       # lệnh chạy lại được nguyên văn
  echo "================================================================"
  echo "=== [$name] $cfg | seed $seed | $(date '+%F %T')"
  echo "================================================================"
  local t0; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU "${cmd[@]}" 2>&1 | tee "$out/console.log"   # console.log: mọi thứ, kể cả traceback
  local rc=${PIPESTATUS[0]}
  local dt=$(( $(date +%s) - t0 ))
  local st=OK; [ "$rc" -ne 0 ] && st="FAIL($rc)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$cfg" "$seed" "$st" "$dt" "$out" >> "$STATUS"
  if [ "$rc" -ne 0 ]; then
    echo "!! LỖI ($rc) ở $name seed $seed — xem $out/console.log (chạy tiếp thí nghiệm khác)"; return 0; fi
  if [ "$KEEP_CKPT" != "1" ] && [ "$name" != "C_wordRO_baseline" ] && [ "$name" != "D_wordRO_LSS" ]; then
    rm -f "$out"/pytorch_model.bin "$out"/model.safetensors                # giữ log, bỏ trọng số cho đỡ tốn đĩa
  fi
}

case "$STAGE" in
  smoke)
    OUT="$OUT/_smoke"; STATUS="$OUT/run_status.tsv"; mkdir -p "$OUT"
    printf 'time\texperiment\tconfig\tseed\tstatus\tseconds\toutput_dir\n' > "$STATUS"
    SMOKE=(--max_steps 20 --max_train_samples 4 --max_test_samples 4 --logging_steps 5)
    run smoke_baseline $MAIN 42 --do_train --do_predict "${SMOKE[@]}"
    run smoke_LSS      $MAIN 42 --do_train --do_predict "${SMOKE[@]}" "${LATENT[@]}"
    cat "$STATUS"
    ;;

  tune)
    # Chỉ dùng DEV (tách cố định dev_split_seed=42). Log dev F1 mỗi 100 bước -> log_history.jsonl
    for L in 4 6 8; do
      run "tune_latent_L$L" $MAIN 42 --do_train --do_eval --dev_ratio 0.1 \
          "$EVAL_FLAG" steps --eval_steps 100 --use_latent_segment True --latent_layer $L
    done
    run tune_baseline $MAIN 42 --do_train --do_eval --dev_ratio 0.1 "$EVAL_FLAG" steps --eval_steps 100
    ;;

  final)
    # KHÔNG đánh giá trên test trong lúc train; chỉ predict 1 lần ở checkpoint cuối.
    for s in "${SEED_ARR[@]}"; do
      run A_goldseg_baseline   funsd         $s --do_train --do_predict              # tham chiếu (rò rỉ nhãn)
      run B_word_baseline      funsd_word    $s --do_train --do_predict
      run C_wordRO_baseline    $MAIN         $s --do_train --do_predict
      run D_wordRO_LSS         $MAIN         $s --do_train --do_predict "${LATENT[@]}" # PHƯƠNG PHÁP CHÍNH
      run E_word_LSS           funsd_word    $s --do_train --do_predict "${LATENT[@]}"
    done
    ;;

  ablation)
    for s in "${SEED_ARR[@]}"; do
      # Cận trên: dùng lại checkpoint D, chỉ đổi sang nhóm gold lúc test (không train lại)
      if [ -d "$OUT/D_wordRO_LSS/seed$s" ]; then
        run F_wordRO_LSS_oracle $MAIN $s --do_predict "${LATENT[@]}" --latent_oracle_eval True \
            --model_name_or_path "$OUT/D_wordRO_LSS/seed$s"
      else
        echo "!! Thiếu checkpoint $OUT/D_wordRO_LSS/seed$s — chạy 'final' trước"; fi
      run G_noAttnBias   $MAIN $s --do_train --do_predict "${LATENT[@]}" --latent_attn_bias False
      run H_noSoftBox    $MAIN $s --do_train --do_predict "${LATENT[@]}" --latent_soft_box False
      run I_noTF         $MAIN $s --do_train --do_predict "${LATENT[@]}" --latent_tf_ratio 0
      run J_noAffLoss    $MAIN $s --do_train --do_predict "${LATENT[@]}" --affinity_loss_weight 0
      run K_noImage_LSS  $MAIN $s --do_train --do_predict "${LATENT[@]}" --visual_embed False
      run K_noImage_base $MAIN $s --do_train --do_predict --visual_embed False
    done
    ;;

  report)
    OUT="$OUT" SEEDS="$SEEDS" python - <<'PY'
import glob, json, os, re
import numpy as np

root, seeds = os.environ["OUT"], os.environ["SEEDS"].split()
# Thí nghiệm -> thí nghiệm tham chiếu để tính delta (cùng setting, cùng seed)
REF = {"B_word_baseline": "A_goldseg_baseline", "C_wordRO_baseline": "B_word_baseline",
       "D_wordRO_LSS": "C_wordRO_baseline", "E_word_LSS": "B_word_baseline",
       "F_wordRO_LSS_oracle": "D_wordRO_LSS", "G_noAttnBias": "D_wordRO_LSS", "H_noSoftBox": "D_wordRO_LSS",
       "I_noTF": "D_wordRO_LSS", "J_noAffLoss": "D_wordRO_LSS", "K_noImage_LSS": "K_noImage_base",
       "K_noImage_base": "C_wordRO_baseline", "tune_latent_L4": "tune_baseline",
       "tune_latent_L6": "tune_baseline", "tune_latent_L8": "tune_baseline"}

def load(exp):
    """-> (split, prefix, {seed: metrics})"""
    for split, fn, pre in (("test", "test_results.json", "test_"), ("dev", "eval_results.json", "eval_")):
        runs = {}
        for f in sorted(glob.glob(os.path.join(root, exp, "seed*", fn))):
            seed = os.path.basename(os.path.dirname(f))[4:]
            runs[seed] = json.load(open(f))
            tr = os.path.join(os.path.dirname(f), "train_results.json")
            if os.path.exists(tr):
                runs[seed].update({"train_" + k.replace("train_", ""): v for k, v in json.load(open(tr)).items()})
        if runs:
            return split, pre, runs
    return None, None, {}

def ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)

exps = sorted(e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e)) and not e.startswith("_"))
data = {e: load(e) for e in exps}
summary, rows = {}, []
for e in exps:
    split, pre, runs = data[e]
    if not runs:
        continue
    keys = sorted({k for r in runs.values() for k, v in r.items() if isinstance(v, (int, float))})
    stats = {k: dict(zip(("mean", "std"), ms([r[k] for r in runs.values() if k in r])),
                     n=sum(k in r for r in runs.values())) for k in keys}
    main_key = pre + "f1"
    ent = {"split": split, "seeds": sorted(runs), "missing_seeds": [s for s in seeds if s not in runs]
           if not e.startswith("tune") else [], "main_key": main_key,
           "main_f1": {**stats[main_key], "values": [runs[s][main_key] for s in sorted(runs)]}, "metrics": stats}
    ref = REF.get(e)
    if ref and data.get(ref, (None, None, {}))[2]:
        rruns = data[ref][2]
        common = sorted(set(runs) & set(rruns))
        if common:                                   # delta theo CẶP seed (khử nhiễu do seed)
            d = [runs[s][main_key] - rruns[s][main_key] for s in common]
            dm, ds = ms(d)
            ent["delta_vs"] = {"ref": ref, "paired_seeds": common, "mean": dm, "std": ds,
                               "wins": int(sum(x > 0 for x in d))}
    summary[e] = ent
    json.dump(ent, open(os.path.join(root, e, "summary.json"), "w"), indent=2, ensure_ascii=False)

# --------- bảng in ra + summary.md ---------
def g(e, k, scale=100, digits=2):
    m = summary[e]["metrics"].get(summary[e]["main_key"].split("f1")[0] + k)
    return f"{scale * m['mean']:.{digits}f}" if m else "-"

types = sorted({re.match(r"(?:test|eval)_([A-Z][A-Z_.]*)_f1$", k).group(1)
                for e in summary for k in summary[e]["metrics"] if re.match(r"(?:test|eval)_([A-Z][A-Z_.]*)_f1$", k)})
types = types if len(types) <= 4 else []           # CORD có nhiều loại -> xem summary.json
hdr = ["experiment", "split", "n", "F1 mean±std", "P", "R"] + [f"F1 {t[:8]}" for t in types] + \
      ["pairF1", "TYPE", "SPAN", "FP", "FN", "Δ vs ref (paired)", "time/run"]
lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for e, s in summary.items():
    m = s["main_f1"]
    d = s.get("delta_vs")
    dtxt = f"{100*d['mean']:+.2f}±{100*d['std']:.2f} vs {d['ref']} ({d['wins']}/{len(d['paired_seeds'])} thắng)" if d else "-"
    t = s["metrics"].get("train_runtime", {}).get("mean")
    row = [e, s["split"], str(m["n"]), f"{100*m['mean']:.2f} ± {100*m['std']:.2f}", g(e, "precision"), g(e, "recall")] + \
          [g(e, f"{ty}_f1") for ty in types] + [g(e, "group_pair_f1")] + \
          [g(e, f"err_{k}", 1, 1) for k in ("TYPE", "SPAN", "FP", "FN")] + \
          [dtxt, f"{t/60:.0f}m" if t else "-"]
    lines.append("| " + " | ".join(row) + " |")
    if s["missing_seeds"]:
        print(f"[thiếu seed] {e}: {s['missing_seeds']}")
    if s["metrics"].get(s["main_key"].replace("f1", "metric_consistent"), {}).get("mean", 1) < 1:
        print(f"[CẢNH BÁO] {e}: F1 đếm lỗi lệch seqeval ở ít nhất 1 seed")

# Phần khoảng hở lấy lại được: (D - C) / (F - C)
if all(k in summary for k in ("C_wordRO_baseline", "D_wordRO_LSS", "F_wordRO_LSS_oracle")):
    c, d_, f = (summary[k]["main_f1"]["mean"] for k in ("C_wordRO_baseline", "D_wordRO_LSS", "F_wordRO_LSS_oracle"))
    if f - c > 1e-9:
        lines.append(f"\nKhoảng hở lấy lại được (D−C)/(F−C) = {100*(d_-c)/(f-c):.1f}%")

table = "\n".join(lines)
print(table)
open(os.path.join(root, "summary.md"), "w").write(
    "Metric chính: seqeval entity-level F1 (IOB2, mode mặc định), cấp tài liệu. "
    "Lỗi = số entity trung bình mỗi seed.\n\n" + table + "\n")
json.dump(summary, open(os.path.join(root, "summary.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nĐã lưu {root}/summary.md, {root}/summary.json, {root}/<exp>/summary.json")
fails = [l for l in open(os.path.join(root, "run_status.tsv")).read().splitlines()[1:] if "FAIL" in l] \
    if os.path.exists(os.path.join(root, "run_status.tsv")) else []
if fails:
    print("\n[LƯỢT LỖI]\n" + "\n".join(fails))
PY
    ;;
  *) echo "Dùng: bash $0 {smoke|tune|final|ablation|report}"; exit 1 ;;
esac