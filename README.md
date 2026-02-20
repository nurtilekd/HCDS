## 0. Hard Constrained Downscaling using SoftMax Function

Compares two neural approaches to **spatial downscaling** of gridded climate fields: a standard SRCNN and an SRCNN augmented with a **SoftMax Constraining Layer (SmCL)** that enforces a hard local conservation constraint on each 2×2 block.

The pipeline is trained and evaluated on 10 years (2001–2010) of ERA5 reanalysis data across 8 near-surface variables, downscaling from 180×360 global grids.

Reference paper: *"Hard-Constrained Deep Learning for Climate Downscaling"* — Paula Harder et al.

---

## 1. Repository Structure

```
HCDS/
├─ runs/
│  ├─ SRCNN_2m_temperature/                                ← Checkpoints, loss curves, example maps
│  └─ SRCNN_SmCL_2m_temperature/                           ← Checkpoints, loss curves, example maps
├─ ranges.py                                               ← Global per-variable min/max bounds
├─ raw_to_labels.ipynb                                     ← Convert NetCDF → NPZ dataset
├─ example_label.ipynb                                     ← Visualise HR vs LR inputs
├─ SRCNN.ipynb                                             ← Train + evaluate unconstrained SRCNN
├─ SRCNN_SmCL.ipynb                                        ← Train + evaluate SRCNN + SmCL
└─ README.md                                               ← Info
```

---

## 2. Variables

Eight ERA5 near-surface variables are used, covering temperature, wind, radiation, humidity, pressure, and precipitation:

| Variable | Global Min | Global Max |
|----------|-----------|-----------|
| `2m_temperature` | 192.02 K | 320.73 K |
| `10m_u_component_of_wind` | −38.02 m/s | 32.33 m/s |
| `10m_v_component_of_wind` | −32.96 m/s | 36.67 m/s |
| `mean_surface_downward_long_wave_radiation_flux` | 40.73 W/m² | 497.72 W/m² |
| `mean_surface_downward_short_wave_radiation_flux` | 0.0 W/m² | 487.31 W/m² |
| `specific_humidity` | 6.4e-7 kg/kg | 0.0313 kg/kg |
| `surface_pressure` | 48,785 Pa | 106,520 Pa |
| `total_precipitation` | 0.0 m | 0.024 m |

Global min/max bounds are tracked incrementally across all years in `ranges.py` and used for normalisation and denormalisation. Note that `mean_surface_downward_short_wave_radiation_flux` receives special treatment during preprocessing — small negative noise artifacts from the ERA5 source are clamped to zero before any further processing.

---

## 3. Approach

**LR input construction:** Each HR field (180×360) is downscaled by reshaping into 2×2 blocks and taking the `nanmean` within each block, producing a 90×180 low-resolution grid. That grid is then tiled back to 180×360 by repeating each value into its 2×2 neighbourhood. The result is a blocky, spatially smeared version of the original — the model's job is to reconstruct the fine-grained structure that was lost. Masked ocean/land values (common in ERA5) are filled with the mean of valid pixels in the same field before this step, rather than left as NaNs that would corrupt gradients.

**Normalisation:** All fields are min-max normalised to [0, 1] using the global bounds in `ranges.py`. Normalisation is applied *after* LR construction, so both `X` (LR-expanded) and `Y` (HR target) share the same scale. The bounds are updated incrementally — each new year's data expands the range if it falls outside the previously observed limits.

**Model — SRCNN:** A 3-layer convolutional network trained end-to-end with Adam (lr = 1e-4, batch size = 32, 400 epochs) and L1 loss:

| Layer | Kernel | Channels | Activation |
|-------|--------|----------|------------|
| Conv2d | 9×9, pad 4 | 1 → 64 | ReLU |
| Conv2d | 1×1 | 64 → 32 | ReLU |
| Conv2d | 5×5, pad 2 | 32 → 1 | — |

The wide 9×9 first kernel captures broad spatial context across the blurry LR input. The 1×1 middle layer acts as a learned feature mixer across the 64 channels. The final 5×5 layer reconstructs the high-frequency output.

**Model — SRCNN + SmCL:** The CNN backbone is identical, but the output is treated as **logits** rather than a direct pixel prediction. The SoftMax Constraining Layer takes those logits and computes SoftMax weights within each 2×2 spatial block (four weights that sum to 1). These weights are then multiplied against the corresponding LR values to produce the final HR prediction.

The constraint is architectural — the model physically cannot predict values outside the convex hull of its LR inputs. It can only decide *how to redistribute* the energy already present in each 2×2 neighbourhood, not *how much* total energy to emit. This is what makes the constraint hard rather than soft.

**Validation:** Data is split 80% train / 10% val / 10% test using a fixed random seed for reproducibility. A naive baseline — using the LR-expanded input directly as the prediction — is evaluated first and stored in `report.json` alongside trained metrics, giving a concrete floor to measure improvement against.

---

## 4. Notebooks

### `raw_to_labels.ipynb`
Converts yearly `.nc` files from `./input_raw/` into training-ready `.npz` pairs in `./input_labels/`. Iterates over all combinations of `YEARS` (2001–2010) and `VARIABLES` (8 ERA5 fields), handles masking and NaN filling, constructs LR inputs, normalises, and saves each `(year, variable)` pair as a compressed `.npz`. Also maintains `ranges.py` — updating global per-variable bounds on the fly as each year is processed.

Each output `.npz` contains:

| Key | Description |
|-----|-------------|
| `X` | LR-expanded input, shape `(T, 1, H, W)`, normalised to [0, 1] |
| `Y` | HR target, shape `(T, 1, H, W)`, normalised to [0, 1] |
| `norm` | Normalisation scheme (`"zero_one"`) |
| `vmin`, `vmax` | Per-file bounds used during this normalisation pass |

![image_alt](images/1.png)

### `ranges.py`
Stores the global min/max bounds observed across all years for each variable. Written and updated automatically by `raw_to_labels.ipynb`. These are the values you'd use to invert normalisation and recover physical units from model predictions.

### `example_label.ipynb`
Loads a single `(year, variable, day)` directly from `./input_raw/`, reconstructs the LR-expanded version, and plots both side-by-side using a shared colour scale. Applies a longitude roll to recentre the globe at 0° and a vertical flip for north-up orientation. Useful for confirming that the 2×2 block averaging and upsampling are working correctly before running the full preprocessing pipeline.

![image_alt](images/2.png)

### `SRCNN.ipynb`
Trains and evaluates the unconstrained SRCNN. Loads the full dataset for the configured `variable` across all `years`, verifies normalisation to [0, 1], and performs an 80/10/10 random split with a fixed seed. Training runs for 400 epochs; the best checkpoint (lowest validation L1) is saved to `./runs/SRCNN_{variable}/best_model.pt`. After training, the best checkpoint is reloaded and evaluated on the test set. Results — baseline error, trained error, and percentage improvement — are written to `report.json`.

![image_alt](images/3.png)

### `SRCNN_SmCL.ipynb`
Identical training and evaluation setup to `SRCNN.ipynb`. The only difference is the final prediction step: the CNN's raw output is passed through the SoftMax Constraining Layer before being compared to the HR target. The backbone weights, optimiser, loss function, and evaluation code are all unchanged, making the comparison between the two notebooks a clean ablation of the constraint alone.

![image_alt](images/4.png)

![image_alt](images/5.png)

---

## 5. Results

Both models substantially outperform the LR-expanded baseline, confirming the networks learn real spatial structure. Adding SmCL reduces MAE and RMSE further for most variables.

| Model | vs. Baseline |
|-------|-------------|
| SRCNN | Large MAE/RMSE improvement across all variables |
| SRCNN + SmCL | Additional reduction over plain SRCNN for most variables |

![image_alt](images/6.png)

The strongest SmCL gains appear in smooth, spatially coherent fields — surface pressure, 2m temperature, 10m wind components, and longwave radiation. These benefit most because the convex-combination constraint naturally preserves large-scale spatial structure while suppressing reconstruction artifacts at block boundaries. Noisier fields like `total_precipitation` and `specific_humidity` show smaller or inconsistent gains, likely because sharp local gradients make the hard constraint more restrictive than helpful.

> Note: all reported errors are in normalised [0, 1] units. To recover physical quantities, apply the inverse min-max transform using the `vmin`/`vmax` stored in `ranges.py`.

These results are consistent with Harder et al. (2024), who report similar gains from SoftMax constraining on ERA5 total column water across multiple task difficulties (4× spatial SR, multi-step temporal SR, joint spatial+temporal SR). That the same pattern holds here across a broader set of near-surface variables suggests hard output constraints are a robust architectural strategy rather than one tuned to a specific variable type.

---

## 6. Workflow

```
./input_raw/          ← Place yearly .nc files here (e.g. 2001.nc … 2010.nc)
       ↓  raw_to_labels.ipynb
./input_labels/       ← Per-variable .npz pairs + updated ranges.py
       ↓  SRCNN.ipynb or SRCNN_SmCL.ipynb
./runs/<run_name>/    ← best_model.pt, loss curve, example maps, report.json
```