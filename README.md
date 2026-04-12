# PIR-JEPA — Physics Law Discovery API

**Physics Intermediate Representation** with Joint Embedding Predictive Architecture
prior and Langevin diffusion sampling.

Submit tabular physics data → receive the discovered symbolic law + hidden physics
sensitivity score (Δs).

## Live Demo

🔬 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Qazihanif/pir-jepa)**

## Papers

| Paper | DOI |
|-------|-----|
| PIR-JEPA v2 (this work) | [10.5281/zenodo.19477508](https://doi.org/10.5281/zenodo.19477508) |
| PIR Architecture v3 | [10.5281/zenodo.19428230](https://doi.org/10.5281/zenodo.19428230) |
| PIR-Bench v3.1 | [10.5281/zenodo.19130521](https://doi.org/10.5281/zenodo.19130521) |
| PhysicsGPT v3 | [10.5281/zenodo.19130163](https://doi.org/10.5281/zenodo.19130163) |

## Key Results (5-seed ablation, April 2026)

### Main result

| Configuration | DR% | MAE | DCS |
|---|---|---|---|
| Baseline (OT only) | 0% | 0.808 | 0.100 |
| Template augmentation only | 80% | 0.008 | 0.885 |
| **PIR-JEPA (Langevin T=500)** | **100%** | **0.008** | **0.994** |

### Langevin step count ablation — non-monotonic DR curve

| Steps | DR% | Regime |
|---|---|---|
| 0 (template only) | 80% | No diffusion |
| 200 | 80% | Under-diffusion |
| **500** | **100%** | **✓ Optimal** |
| **750** | **100%** | **✓ Optimal** |
| 900 | 80% | Over-diffusion begins |
| 1000 | 60% | Over-diffusion |

**Over-diffusion phenomenon:** beyond 750 steps the Langevin sampler walks past
the v|v| basin of attraction in latent space, reducing DR. This is the first
reported instance of over-diffusion in score-based symbolic expression generation.
Optimal window: **500–750 steps**.

## Architecture

PIR-JEPA uses two additive candidate sources beyond the standard grammar:

**1. Template augmentation** — six physics-informed expression families:
- Compound-angle trigonometry
- Nonlinear damping: v·|v|, v², |v|
- Exponential / Boltzmann forms
- Logarithmic patterns
- Non-integer power-law cross terms

**2. Langevin diffusion** — score-based walk on the JEPA physics manifold:
```
z_{t+1} = z_t + (η/2)·score(z_t) + √η·ε,   ε ~ N(0, σ²I)
score(z) = -(z - predictor(encoder(z)))
```
T=500 steps, η=0.01, σ=0.1.

**Scoring:**
```
s_total(c) = s_OT(c) + 0.2·s_JEPA(c)
s_OT(c)    = -(0.7·MSE + 0.3·W₁)
```

## Quick Start

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860/docs` for the interactive Swagger UI.

## API

### POST /discover

```json
{
  "data": [[0.1, 0.5, -0.35], [-0.3, 1.2, 0.67], ...],
  "variables": ["x", "v"],
  "target": "F",
  "use_jepa": true,
  "langevin_steps": 500
}
```

**Response:**
```json
{
  "expression": "-1.0*x - 0.5*v*Abs(v)",
  "mae": 0.0082,
  "dcs": 0.994,
  "delta_s": 0.107,
  "rank2_expression": "-1.0*x - 0.48*v*Abs(v) + 0.02*log(Abs(v)+1)",
  "hidden_physics_flag": true,
  "langevin_steps_used": 500,
  "diffusion_regime": "optimal",
  "runtime_seconds": 1.3,
  "jepa_active": true
}
```

### GET /ablation

Returns the confirmed Langevin step count ablation results.

### GET /example

Returns a ready-to-use request body for `oog_damped_oscillator`.

### GET /health

Returns engine status, version, and optimal step count range.

## Hidden Physics Sensitivity

**Δs = s(rank-1) − s(rank-2)**

Small Δs < 0.15 flags the dataset as a hidden physics candidate — the rank-2
Langevin expression is a plausible symbolic correction to the dominant law,
consistent with dimensional analysis.

Applied to Kepler's law: the Langevin sampler generates a dense catalogue of
near-Kepler modifications (r^(3/2+δ), r^(3/2)·log r) corresponding to dark matter,
fifth force, and extra dimension corrections. Longer JEPA runtime = denser catalogue
= higher hidden physics sensitivity.

## Benchmark Datasets

```bash
python bench/run_discovery_benchmark.py \
    --experiments oog_damped_oscillator \
    --noise-levels 0.01 --dataset-sizes 200 \
    --repeats 5 --hybrid-ot --use-jepa \
    --langevin-steps 500
```

## Citation

```bibtex
@misc{hanif2026pirjepa,
  author    = {Muhammad Hanif},
  title     = {{PIR-JEPA}: Joint Embedding Predictive Architecture
               as a Physics Manifold Prior for
               Out-of-Grammar Symbolic Law Discovery},
  year      = {2026},
  doi       = {10.5281/zenodo.19477508},
  publisher = {Zenodo}
}
```

## Note on Code Availability

The core inference engine is proprietary. This repository provides:
- The public API wrapper (`api.py`)
- Interactive demo UI (`index.html`)
- Benchmark datasets and evaluation scripts (`bench/`)
- Confirmed ablation results via `/ablation` endpoint

This satisfies arXiv reproducibility standards. The live demo at
[huggingface.co/spaces/Qazihanif/pir-jepa](https://huggingface.co/spaces/Qazihanif/pir-jepa)
provides interactive verification.

For collaboration or licensing: qmhanif70@gmail.com
