# Forest plot scripts

Renders the per-intent forest plots (one row per model: GPT-4, GPT-3.5,
Llama2-70b, Llama2-13b; a "Low / A / High" band marking the human reference
range) for three analyses:

- **Frequency** — how often each intent occurs
- **Temporal order** — where in the conversation each intent tends to occur
- **Client adaptability** — how often each therapist intent follows each
  client intent

## Files

| File | Purpose |
|---|---|
| `plot_graph_common.py` | Shared rendering engine (style, ticks, colors, the forest-plot renderer). Not run directly. |
| `plot_graph_frequency.py` | Frequency chart, one PNG per intent |
| `plot_graph_temporal_order.py` | Temporal-order chart, one PNG per intent |
| `plot_graph_client_adaptability.py` | Client-adaptability chart, one PNG per (client intent, therapist intent) pair |

## Input

Each script reads exactly one CSV from `data/` — nothing else:

| Script | Reads |
|---|---|
| `plot_graph_frequency.py` | `data/all_dist.csv` |
| `plot_graph_temporal_order.py` | `data/all_order.csv` |
| `plot_graph_client_adaptability.py` | `data/all_dist_clientvst.csv` |

These are per-model **significance tables**, one row per (model, intent)
combination (or per (model, therapist intent, client intent) combination for
the adaptability one), trimmed to just the columns the scripts use. Each
already carries everything the chart needs:

- `model_name` — either `{model}_all` (e.g. `gpt4_all`) or one of
  `human_avg` / `human_low` / `human_high` (the human reference band, whose
  value is in their own `avg` column)
- `avg`, `std`, `size_for_model_name` — the model's mean, standard
  deviation, and sample size for that intent
- `intent` (or `intent_therapist` + `intent_client` for the adaptability
  table) — which intent/pair this row is
- `pvalue` — a boolean, not a raw p-value: the multiple-comparison-corrected
  significant/not-significant flag that decides whether a model's dot is
  colored or shown grey (originally `new_pvalue_identify` upstream; renamed
  here since the raw and corrected numeric p-values aren't used and were
  dropped)

No raw per-conversation data and no separate "relative diff" table are
needed — the scripts derive the dot's color direction and the error bar
directly from these columns.

The "Low"/"A"/"High" labels are positioned automatically — `plot_graph_common.py`
detects when they'd overlap (which happens whenever the human range is narrow
relative to how far the farthest model sits from it) and spreads them apart,
so no per-intent manual tuning is needed.

## Output

Each script creates its own subfolder under `output/` and writes one PNG per
chart:

```
output/
  frequency/{intent}.png
  temporal_order/{intent}.png
  client_adaptability/{client_intent}_{therapist_intent}.png
```


## Running

Requires `pandas`, `matplotlib`, `seaborn`, `scipy`, `numpy` (tested with
pandas 2.2, matplotlib 3.8, seaborn 0.13, scipy 1.13, numpy 1.26 — seaborn
0.13+ is what `plot_graph_common.py` targets; nothing here relies on the
`sb.pointplot` internals of older seaborn versions since a manual
`errorbar` is used instead). Run each script from inside this folder:

```bash
cd analysis_code
python plot_graph_frequency.py
python plot_graph_temporal_order.py
python plot_graph_client_adaptability.py
```

No command-line arguments — edit the `SIG_CSV_PATH`/`SAVE_DIR_GRAPH`
constants at the top of a script to point it elsewhere.