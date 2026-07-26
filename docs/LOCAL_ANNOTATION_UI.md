# Local OOP Annotation UI

This tool lets an annotator review the exported OOP/orientation crop pack on the local laptop. It does not upload images, call network services, change pipeline outputs, or compute validation statistics.

## Start

From `<repo-root>`:

```bash
../sarcgraph-env/bin/python scripts/annotate_oop_pack.py --config configs/default.yaml
```

Before opening the interactive window, you can check that the pack and crop paths are present:

```bash
../sarcgraph-env/bin/python scripts/annotate_oop_pack.py --config configs/default.yaml --headless-check
```

The UI reads:

- `results/annotation_pack/annotation_patch_index.csv`
- `results/annotation_pack/annotation_template.csv`
- `results/annotation_pack/crops/`

It writes:

- `results/annotation_pack/annotation_filled.csv`
- `results/annotation_pack/annotation_filled.autosave.csv`

If `annotation_filled.csv` already exists, the UI resumes from it. Use `--overwrite` only when you intentionally want to reinitialize the annotation sheet from the template.

## Fields

For each crop, annotate:

- `manual_dominant_orientation_deg`: axial orientation from 0 to 180 degrees. Use blank/NaN if no clear orientation is measurable.
- `manual_organisation_score`: 1 to 5.
- `manual_organisation_label`: optional text label.
- `visible_striations_yes_no`: `yes`, `no`, or `yes_unclear`.
- `confidence_score`: 1 to 5.
- `notes`: free text.

The sarcomere length field is optional and remains blank by default. Spacing is exploratory/low-yield in the production pipeline, so do not force spacing measurements in this OOP annotation pass.

## Angle Convention

Orientation is axial:

- 0 degrees = horizontal.
- 90 degrees = vertical.
- 180 degrees is equivalent to 0 degrees.

Use NaN/blank for no clear dominant orientation.

## Organisation Score

1 = disorganised / no coherent striation orientation  
2 = weakly organised  
3 = moderately organised  
4 = strongly organised  
5 = highly organised

Annotate all exported crops, including low-quality or visually ugly examples. Skipping difficult crops would introduce selection bias.

## Keyboard Shortcuts

- `1`-`5`: set organisation score and default label.
- `y`: visible striations yes.
- `u`: visible striations unclear.
- `n`: visible striations no.
- `r`: mark manual orientation as NaN / not measurable.
- `enter`: prompt in the terminal for angle/optional text, save, and advance.
- `b`: go back.
- `s`: save.
- `q`: save and quit.

Autosave is written after each crop update.
