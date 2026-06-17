#A Zero-Watermarking Method for Vector Geographic Data Based onSparse Mixture of Experts

Official code for the paper (V17.12 — standard recovered-watermark NC evaluation).

This repository contains the implementation of a **Sparse Mixture-of-Experts (MoE) zero-watermarking system** for remote-sensing / map vector data, featuring:

- **GeoVecFormer-ZW**: a multi-branch feature extractor that fuses raster grid, vector-token, and topology-graph representations.
- **Sparse MoE RouterNet**: geometry-descriptor-based routing that activates top-K specialist experts per sample.
- **Standard zero-watermark pipeline**: copyright registration (`Z = W ⊕ B`) and recovery (`W' = Z ⊕ B'`), evaluated with bitwise NC (normalized correlation = 1 − BER).

---

## Repository Structure

```
rb_afl_system/
├── data/
│   ├── channels/        # Multi-channel raster builder
│   ├── dataset/         # Triplet dataset for contrastive training
│   ├── features/        # Vector-token & topology-graph feature extractors
│   └── geometry/        # Geometry utility functions
├── engine/
│   ├── checkpoint_io.py # Model checkpoint save/load
│   └── device.py        # Device selection helper
├── models/
│   ├── baseline/        # CNN and ResNet-SE generator baselines; FC discriminators
│   └── geovecformer/    # GeoVecFormer-ZW: grid + token + graph fusion model
│       ├── branch_generators.py
│       ├── fusion.py                       # Gated multi-branch fusion
│       ├── geovecformer_zw.py              # Main model
│       ├── graph_transformer_encoder.py
│       ├── grid_encoder.py
│       ├── topology_specialist_generators.py
│       └── vector_token_transformer.py
├── router/
│   └── geometry_descriptor.py  # Lightweight geometry features for MoE routing
├── scripts/
│   ├── moe_ablation_paper_nc_V17_12.py          # Main V17.12 ablation entry
│   ├── standard_paper_nc_moe_eval_V17_6.py      # Standard paper-NC core functions
│   ├── sparse_moe_router_V16.py                 # RouterNet (V16)
│   ├── sparse_moe_router_V16_1.py               # RouterNet (V16.1, with protected roles)
│   ├── sparse_moe_fusion_train_V17.py           # MoE fusion trainer
│   ├── sparse_moe_torch_fusion_train_V17_1.py   # MoE fusion trainer (torch gated)
│   ├── specialist_ensemble_evaluator_V13.py     # Specialist ensemble evaluator
│   └── specialist_ensemble_evaluator_V15.py     # Specialist ensemble evaluator (V15)
├── watermark/
│   ├── zero_watermark.py  # Zero-watermark register / recover / evaluate
│   ├── metrics.py         # NC and BER metrics
│   └── feature_to_bits.py # Feature vector → binary bits
└── utils.py               # Common utilities (seed, I/O, logging)

run_eval.sh        # Unified evaluation entry point
requirements.txt   # Python dependencies
```

---

## Zero-Watermark Protocol

The standard evaluation metric used in this paper is the **recovered-watermark NC**:

```
Registration:  Z  = W  ⊕ B          (copyright bits W, feature bits B)
Recovery:      W' = Z  ⊕ B'         (B' = feature bits under attack)
Evaluation:    NC = mean(W == W') = 1 − BER
```

This is implemented in `rb_afl_system/watermark/zero_watermark.py`.

---

## Environment Setup

Python ≥ 3.9 and PyTorch ≥ 1.13 are recommended.

```bash
conda create -n rbafl python=3.10
conda activate rbafl
pip install -r requirements.txt
```

---

## Running the Evaluation

The evaluation script runs the full V17.12 MoE ablation paper-NC re-evaluation, covering:
- Main split MoE ablation
- Repeated-split stability test
- Mixed zero-shot generalization
- Attack-strength sweep

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `RB_ROOT` | Root directory of your experiment data | `/home/user/rb_afl_research` |
| `BASE_RUN` | Path to the V17.1 MoE base run folder | `$RB_ROOT/v17_runs/<run_name>` |
| `MAIN_EXPERT_TABLE` | Path to `sparse_moe_expert_table_paired_v17_1.csv` | auto under `BASE_RUN` |
| `MAIN_VARIANT_ROOT` | Path to `torch_moe_variants/` | auto under `BASE_RUN` |
| `REPEATED_ROOT` | Path to `repeated_splits_v17_1_torch_moe/` | auto under `BASE_RUN` |
| `MIXED_EXPERT_TABLE` | Path to mixed zero-shot expert table (optional) | auto-searched |
| `ATTACK_STRENGTH_EXPERT_TABLE` | Path to attack-strength sweep table (optional) | auto-searched |
| `OUT` | Output directory | `$RB_ROOT/v17_12_runs/final_paper_nc_clean_<timestamp>` |

### Run

```bash
export RB_ROOT=/path/to/your/experiment/data
bash run_eval.sh 2>&1 | tee run_final.log
```

The script will:
1. Check code syntax and integrity.
2. Run the full V17.12 paper-NC evaluation.
3. Print key summary tables.
4. Package results into `final_v17_12_paper_nc_result_pack.zip`.

---

## Model Architecture

### GeoVecFormerZW

A multi-branch feature generator fusing three complementary representations:

| Branch | Input | Encoder |
|---|---|---|
| Grid | Raster grid (H×W×C) | `GridEncoder` (CNN) |
| Token | Vector tokens | `VectorTokenTransformer` (Transformer) |
| Graph | Topology graph (nodes + adjacency) | `GraphTransformerEncoder` |

The three branch embeddings are combined by `GatedFusion` into a single feature vector used as the zero-watermark feature `B`.

### Sparse MoE Router

A lightweight `RouterNet` computes geometry descriptors from the input sample and routes it to the top-K specialists. The router is trained to maximize watermark NC while maintaining expert specialization via load-balancing.

---

## Key Metrics

| Metric | Description |
|---|---|
| `paper_wm_xnor_nc__role` | Per-role recovered-watermark NC (= 1 − BER) |
| `router_mean_nc` | Mean NC across all router-selected expert roles |
| `router_min_nc` | Minimum NC across roles (robustness lower bound) |
| `router_nc_lt_0_9` | Fraction of roles with NC < 0.9 |

---

## Requirements

```
numpy
pandas
torch
Pillow
shapely
```

---

## Citation

If you use this code, please cite our paper:

```bibtex
A Zero-Watermarking Method for Vector Geographic Data Based onSparse Mixture of Experts
  title   = {A Zero-Watermarking Method for Vector Geographic Data Based onSparse Mixture of Experts},
  author  = {Li ming Gao},
  year    = {2026},
}
```

---

## License

This project is released under the [MIT License](LICENSE).
