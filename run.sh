#!/usr/bin/env bash
# =============================================================================
# run_latent_segment.sh  — đặt ở thư mục gốc repo (cùng cấp examples/ và layoutlmft/)
#
#   bash run_latent_segment.sh tune      # B1: chọn latent_layer trên DEV (10% train), KHÔNG đụng test
#   bash run_latent_segment.sh final     # B2: bảng chính, 5 seed, số bước cố định, chấm trên test
#   bash run_latent_segment.sh ablation  # B3: ablation + cận trên oracle
#   bash run_latent_segment.sh report    # in mean ± std
#
# Biến môi trường tùy chỉnh: GPU=0 SEEDS="42 43 44 45 46" BS=2 ACC=8 LAYER=6 OUT=runs
# =============================================================================
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GPU=${GPU:-0}
SEEDS=${SEEDS:-"42 43 44 45 46"}
MODEL=${MODEL:-microsoft/layoutlmv3-base}
OUT=${OUT:-runs}
STEPS=${STEPS:-1000}      # giống LayoutLMv3 gốc trên FUNSD
LR=${LR:-1e-5}
BS=${BS:-2}               # BS*ACC = 16 = batch hiệu dụng của paper (8 GPU x 2)
ACC=${ACC:-8}
LAYER=${LAYER:-6}         # cập nhật sau bước tune
MAIN=funsd_word_ro        # SETTING CHÍNH: box từ + thứ tự đọc heuristic, không nhóm gold lúc test
STAGE=${1:-final}

COMMON=(--model_name_or_path "$MODEL" --dataset_name funsd
        --max_steps "$STEPS" --learning_rate "$LR"
        --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$ACC"
        --per_device_eval_batch_size 4 --fp16
        --save_strategy no --logging_steps 50 --report_to none
        --overwrite_output_dir --dataloader_num_workers 4 --input_size 224)

LATENT=(--use_latent_segment True --latent_layer "$LAYER")

# run <tên_thí_nghiệm> <dataset_config> <seed> [tham số thêm...]
run () {
  local name=$1 cfg=$2 seed=$3; shift 3
  local out="$OUT/$name/seed$seed"
  if [ -f "$out/test_results.json" ] || [ -f "$out/eval_results.json" ]; then
    echo ">> bỏ qua (đã có kết quả): $out"; return; fi
  mkdir -p "$OUT/$name"
  echo ">> $name | $cfg | seed $seed"
  CUDA_VISIBLE_DEVICES=$GPU python examples/run_funsd_cord.py "${COMMON[@]}" \
      --dataset_config_name "$cfg" --seed "$seed" --output_dir "$out" "$@" \
      2>&1 | tee "$OUT/$name/seed$seed.log"
}

case "$STAGE" in
  tune)
    # Chỉ dùng DEV. Seed train 42, dev tách cố định bằng dev_split_seed=42.
    for L in 4 6 8; do
      run "tune_latent_L$L" $MAIN 42 --do_train --do_eval --dev_ratio 0.1 \
          --use_latent_segment True --latent_layer $L
    done
    run "tune_baseline" $MAIN 42 --do_train --do_eval --dev_ratio 0.1
    ;;

  final)
    for s in $SEEDS; do
      run A_goldseg_baseline   funsd         $s --do_train --do_predict              # tham chiếu (rò rỉ nhãn)
      run B_word_baseline      funsd_word    $s --do_train --do_predict
      run C_wordRO_baseline    $MAIN         $s --do_train --do_predict
      run D_wordRO_LSS         $MAIN         $s --do_train --do_predict "${LATENT[@]}" # PHƯƠNG PHÁP CHÍNH
      run E_word_LSS           funsd_word    $s --do_train --do_predict "${LATENT[@]}"
    done
    ;;

  ablation)
    for s in $SEEDS; do
      # Cận trên: dùng lại checkpoint D, chỉ đổi sang nhóm gold lúc test (không train lại)
      run F_wordRO_LSS_oracle  $MAIN $s --do_predict --use_latent_segment True --latent_layer "$LAYER" \
          --latent_oracle_eval True --model_name_or_path "$OUT/D_wordRO_LSS/seed$s"
      run G_noAttnBias  $MAIN $s --do_train --do_predict "${LATENT[@]}" --latent_attn_bias False
      run H_noSoftBox   $MAIN $s --do_train --do_predict "${LATENT[@]}" --latent_soft_box False
      run I_noTF        $MAIN $s --do_train --do_predict "${LATENT[@]}" --latent_tf_ratio 0
      run J_noAffLoss   $MAIN $s --do_train --do_predict "${LATENT[@]}" --affinity_loss_weight 0
      run K_noImage_LSS $MAIN $s --do_train --do_predict "${LATENT[@]}" --visual_embed False
      run K_noImage_base $MAIN $s --do_train --do_predict --visual_embed False
    done
    ;;

  report)
    python - "$OUT" <<'EOF'
import glob, json, os, statistics as st, sys
root = sys.argv[1]
print(f"{'experiment':28s} {'split':5s} {'n':>2s}  {'F1 (mean ± std)':>18s}  {'P':>6s}  {'R':>6s}")
for exp in sorted(os.listdir(root)):
    for split, fname, key in [("test", "test_results.json", "test_"), ("dev", "eval_results.json", "eval_")]:
        fs = sorted(glob.glob(os.path.join(root, exp, "seed*", fname)))
        if not fs:
            continue
        rs = [json.load(open(f)) for f in fs]
        f1 = [r[key + "f1"] * 100 for r in rs]
        p = st.mean(r[key + "precision"] * 100 for r in rs)
        rc = st.mean(r[key + "recall"] * 100 for r in rs)
        sd = st.stdev(f1) if len(f1) > 1 else 0.0
        print(f"{exp:28s} {split:5s} {len(f1):2d}  {st.mean(f1):9.2f} ± {sd:5.2f}  {p:6.2f}  {rc:6.2f}")
EOF
    ;;
  *) echo "Dùng: bash $0 {tune|final|ablation|report}"; exit 1 ;;
esac