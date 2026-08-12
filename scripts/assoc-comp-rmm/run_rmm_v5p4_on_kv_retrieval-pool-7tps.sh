#!/bin/bash
set -e

# v5p4: v5p3 + shifted READ (segment s reads segment s-1's mem). Fixes the
# same-segment future-token leak in any segment containing labels (notably QT).
# Per-layer order: ATTN -> COMPRESS -> GDN -> DECOMPRESS(prev_mem) -> residual.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── resources ──────────────────────────────────────────────────────────────
NP=${NP:-1}
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(( TBS / (PER_DEVICE_BATCH_SIZE * NP) ))

# ── model ──────────────────────────────────────────────────────────────────
L=4
H=4
D=128
BASE_MODEL=llama
FLA_LAYER=GatedDeltaNet
EXPAND_V=2.0
CONV_KERNEL=4

# ── data ───────────────────────────────────────────────────────────────────
K=2
V=2
TOKENIZER_PATH="./tokenizers/kv_alphabet_62/"

# ── training ───────────────────────────────────────────────────────────────
ITERS=200000
WARMUP=10000
EVAL_STEPS=500
EARLY_STOP=500

# ── ClearML ────────────────────────────────────────────────────────────────
# Configure credentials once with `clearml-init`, or provide the standard
# CLEARML_API_* environment variables. Set CLEARML_PROJECT="" to disable.
CLEARML_PROJECT=${CLEARML_PROJECT-"compressing-associations/rmm-v5p4-7tps"}
CLEARML_TAGS=${CLEARML_TAGS:-"rmm,v5p4,kv-retrieval,7tps"}
CLEARML_OUTPUT_URI=${CLEARML_OUTPUT_URI:-}
CLEARML_OUTPUT_ARGS=()
if [ -n "$CLEARML_OUTPUT_URI" ]; then
  CLEARML_OUTPUT_ARGS=(--clearml_output_uri "$CLEARML_OUTPUT_URI")
fi

# ── RMM v5p4 memory path ───────────────────────────────────────────────────
WRITE_MODE=pool
READ_MODE=unpool
USE_PARALLEL_PREFILL=True

# ── sweep ──────────────────────────────────────────────────────────────────
for NUM_MEMORY_VECTORS in 2 1; do
  for N in 1 2; do
    for N_PAIRS in 8 4; do
      for TOKENS_PER_SEGMENT in 7; do
        for STATE_SIZE in 32; do
          for LR in 3e-04 1e-04; do

            DATA_PATH="N${N_PAIRS}-K${K}V${V}-V62_1M"
            PAIRS_PER_SEGMENT=$((TOKENS_PER_SEGMENT / 7))

            RUN_NAME="rmmv5p4_${FLA_LAYER}_${BASE_MODEL}_L${L}H${H}D${D}"
            RUN_NAME="${RUN_NAME}_ss${STATE_SIZE}_M${NUM_MEMORY_VECTORS}_${READ_MODE}_${WRITE_MODE}"
            RUN_NAME="${RUN_NAME}_lr${LR}_bs${TBS}_pps${PAIRS_PER_SEGMENT}_tps${TOKENS_PER_SEGMENT}"

            EXP_PATH="./runs-rmmv5p4/${DATA_PATH}/${RUN_NAME}/run_${N}"
            if [ -d "$EXP_PATH" ]; then
              echo "Skipping existing: $EXP_PATH"
              continue
            fi

            echo "Launching: $EXP_PATH"
            accelerate launch \
              --main_process_port 0 \
              --num_processes $NP \
              --mixed_precision bf16 \
              --config_file accelerate.yaml \
              run_rmm_on_kv_retrieval-v5p4.py \
              --exp_path                    "$EXP_PATH" \
              --per_device_batch_size       $PER_DEVICE_BATCH_SIZE \
              --gradient_accumulation_steps $GRAD_ACC_STEPS \
              --total_batch_size            $TBS \
              --data_path                   "./data/${DATA_PATH}" \
              --tokenizer_path              "$TOKENIZER_PATH" \
              --base_model                  $BASE_MODEL \
              --n_layer                     $L \
              --n_head                      $H \
              --n_embd                      $D \
              --fla_layer                   $FLA_LAYER \
              --state_size                  $STATE_SIZE \
              --expand_v                    $EXPAND_V \
              --conv_kernel                 $CONV_KERNEL \
              --num_memory_vectors          $NUM_MEMORY_VECTORS \
              --write_mode                  $WRITE_MODE \
              --read_mode                   $READ_MODE \
              --use_parallel_prefill        $USE_PARALLEL_PREFILL \
              --tokens_per_segment          $TOKENS_PER_SEGMENT \
              --n_pairs                     $N_PAIRS \
              --n_keys                      $K \
              --n_values                    $V \
              --learning_rate               $LR \
              --max_steps                   $ITERS \
              --warmup_steps                $WARMUP \
              --eval_steps                  $EVAL_STEPS \
              --logging_steps               $EVAL_STEPS \
              --early_stopping_patience     $EARLY_STOP \
              --seed                        $((142 + N)) \
              --clearml_project             "$CLEARML_PROJECT" \
              --clearml_task_name           "${RUN_NAME}-run_${N}" \
              --clearml_tags                "$CLEARML_TAGS" \
              "${CLEARML_OUTPUT_ARGS[@]}"
          done
        done
      done
    done
  done
done

echo "Done"
