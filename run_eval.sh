#!/usr/bin/env bash
set -euo pipefail

# Final cleaned entry for V17.12 standard recovered-watermark paper-NC evaluation.
# This script assumes the existing V17.1 expert tables and RouterNet checkpoints are already on the server.

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RB_ROOT="${RB_ROOT:-${RB_ROOT:-/path/to/your/experiment/data}}"
export BASE_RUN="${BASE_RUN:-$RB_ROOT/v17_runs/rawshp_all_v17_1_sparse_moe_20260506_145440}"

export MAIN_EXPERT_TABLE="${MAIN_EXPERT_TABLE:-$BASE_RUN/sparse_moe_torch_fusion_v17_1_gated/sparse_moe_expert_table_paired_v17_1.csv}"
export MAIN_VARIANT_ROOT="${MAIN_VARIANT_ROOT:-$BASE_RUN/sparse_moe_torch_fusion_v17_1_gated/torch_moe_variants}"
export REPEATED_ROOT="${REPEATED_ROOT:-$BASE_RUN/repeated_splits_v17_1_torch_moe}"

export MIXED_EXPERT_TABLE="${MIXED_EXPERT_TABLE:-$(find "$RB_ROOT" -path '*/mixed_sparse_moe_table/sparse_moe_expert_table_paired_v17_2.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)}"
export ATTACK_STRENGTH_EXPERT_TABLE="${ATTACK_STRENGTH_EXPERT_TABLE:-$(find "$RB_ROOT" -path '*/attack_strength_zero_shot_eval/mixed_oracle_static_eval/sparse_moe_oracle_rows_v16.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)}"

export OUT="${OUT:-$RB_ROOT/v17_12_runs/final_paper_nc_clean_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$CODE"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$CODE"
mkdir -p "$OUT"

echo "============================================================"
echo "===== FINAL V17.12 STANDARD RECOVERED-WATERMARK PAPER NC ==="
echo "============================================================"
echo "[CODE]                         $CODE"
echo "[BASE_RUN]                     $BASE_RUN"
echo "[MAIN_EXPERT_TABLE]            $MAIN_EXPERT_TABLE"
echo "[MAIN_VARIANT_ROOT]            $MAIN_VARIANT_ROOT"
echo "[REPEATED_ROOT]                $REPEATED_ROOT"
echo "[MIXED_EXPERT_TABLE]           ${MIXED_EXPERT_TABLE:-<not found>}"
echo "[ATTACK_STRENGTH_EXPERT_TABLE] ${ATTACK_STRENGTH_EXPERT_TABLE:-<not found>}"
echo "[OUT]                          $OUT"

test -f rb_afl_system/scripts/moe_ablation_paper_nc_V17_12.py
test -f rb_afl_system/scripts/standard_paper_nc_moe_eval_V17_6.py
test -f "$MAIN_EXPERT_TABLE"
test -d "$MAIN_VARIANT_ROOT"

echo "========== 1. code self-check =========="
python -m py_compile \
  rb_afl_system/scripts/moe_ablation_paper_nc_V17_12.py \
  rb_afl_system/scripts/standard_paper_nc_moe_eval_V17_6.py \
  rb_afl_system/scripts/sparse_moe_router_V16.py \
  rb_afl_system/scripts/sparse_moe_router_V16_1.py \
  rb_afl_system/scripts/sparse_moe_fusion_train_V17.py \
  rb_afl_system/scripts/sparse_moe_torch_fusion_train_V17_1.py
python -m tabnanny -v \
  rb_afl_system/scripts/moe_ablation_paper_nc_V17_12.py \
  rb_afl_system/scripts/standard_paper_nc_moe_eval_V17_6.py \
  rb_afl_system/scripts/sparse_moe_router_V16.py \
  rb_afl_system/scripts/sparse_moe_router_V16_1.py \
  rb_afl_system/scripts/sparse_moe_fusion_train_V17.py \
  rb_afl_system/scripts/sparse_moe_torch_fusion_train_V17_1.py
if grep -RInP $'\t' rb_afl_system --include='*.py'; then
  echo "[FATAL] Tab character found"
  exit 1
else
  echo "[OK] no Tab characters"
fi
python - <<'PY'
import ast
from pathlib import Path
files = [
    Path('rb_afl_system/scripts/moe_ablation_paper_nc_V17_12.py'),
    Path('rb_afl_system/scripts/standard_paper_nc_moe_eval_V17_6.py'),
    Path('rb_afl_system/scripts/sparse_moe_router_V16.py'),
    Path('rb_afl_system/scripts/sparse_moe_router_V16_1.py'),
    Path('rb_afl_system/scripts/sparse_moe_fusion_train_V17.py'),
    Path('rb_afl_system/scripts/sparse_moe_torch_fusion_train_V17_1.py'),
]
for fn in files:
    tree = ast.parse(fn.read_text(encoding='utf-8'))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            p = parents.get(node)
            while p is not None:
                if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    raise SystemExit(f"[FATAL] nested def: {fn}:{node.lineno}:{node.name}")
                p = parents.get(p)
print('[OK] no nested def in checked scripts')
PY

echo "========== 2. run final paper-NC contexts =========="
python -m rb_afl_system.scripts.moe_ablation_paper_nc_V17_12 \
  --main_expert_table "$MAIN_EXPERT_TABLE" \
  --main_variant_root "$MAIN_VARIANT_ROOT" \
  --repeated_root "$REPEATED_ROOT" \
  --mixed_expert_table "${MIXED_EXPERT_TABLE:-}" \
  --attack_strength_expert_table "${ATTACK_STRENGTH_EXPERT_TABLE:-}" \
  --output_dir "$OUT/moe_ablation_paper_nc" \
  --device cpu \
  2>&1 | tee "$OUT/01_final_v17_12_paper_nc.log"

echo "========== 3. print key summaries =========="
python - <<'PY'
import os
from pathlib import Path
import pandas as pd
out = Path(os.environ['OUT']) / 'moe_ablation_paper_nc'
paths = [
    out / 'main_split_moe_ablation' / 'moe_ablation_variant_summary_paper_nc_v17_12.csv',
    out / 'mixed_zero_shot_paper_nc' / 'moe_ablation_variant_summary_paper_nc_v17_12.csv',
    out / 'attack_strength_sweep_paper_nc' / 'moe_ablation_variant_summary_paper_nc_v17_12.csv',
]
for p in paths:
    if not p.is_file():
        continue
    df = pd.read_csv(p)
    cols = [c for c in [
        'variant_name','descriptor_type','top_k','fusion_mode','protected_roles',
        'router_mean_nc','router_min_nc','router_nc_lt_0_9','router_nc_lt_0_8'
    ] if c in df.columns]
    print(f"\n===== {p.parent.name} =====")
    print(df[cols].to_string(index=False))
rep = out / 'repeated_variant_stats_paper_nc_v17_12.csv'
if rep.is_file():
    print('\n===== repeated_variant_stats_paper_nc_v17_12.csv =====')
    print(pd.read_csv(rep, header=[0, 1]).head(30).to_string(index=False))
PY

echo "========== 4. package necessary result files =========="
cd "$OUT"
zip -r final_v17_12_paper_nc_result_pack.zip \
  moe_ablation_paper_nc \
  01_final_v17_12_paper_nc.log \
  -x "*.pt" "*.npy" "*.npz" "*/grid.npy" "*/tokens.npz" "*/graph.npz" "*/vector.gpkg" "*/vector.shp" "*/vector.dbf" "*/vector.shx" "*/vector.prj"

echo "============================================================"
echo "DONE"
echo "[RESULT_PACK] $OUT/final_v17_12_paper_nc_result_pack.zip"
ls -lh "$OUT/final_v17_12_paper_nc_result_pack.zip"
