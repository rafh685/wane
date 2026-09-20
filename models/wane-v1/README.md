# Wane exterior concept — revision 03

This revision follows the user's XROS reference image: a plain body, a removable
top cartridge, a black plastic mouthpiece, and smoky grey transparent plastic
around the two visual chambers. The previous lower-body window, rear panel,
ribs, and procedural surface texture have been removed.

The physical-product reference checked was Vaporesso's official XROS page:
https://www.vaporesso.com/series-product/xros-series/xros
It documents a visible transparent pod and PCTG pod material. This is a visual
reference, not a specification for Wane's internal construction.

## Files

- `wane-studio.blend`: editable model, materials, camera and studio lights.
- `wane-concept.glb`: interactive preview, including a static illuminated arc.
- `wane-integration.glb`: export without that arc, for later website integration.
- `build_model.py`: reproducible Blender source.
- `index.html`: rotatable preview with finish controls and Lift / Reseat pod.
- `archive-before-top-pod/`: prior script, model and Blender file.

The preview is served locally from this folder on port 8842.

## Animation grouping

`POD_REMOVABLE_ASSEMBLY` contains `WN-02_DUAL_POD` and `WN-01_MOUTHPIECE`.
Move their common parent to lift the entire cartridge. The visible chamber
volumes, divider, menisci and casing follow the pod. The LED remains attached
to the main body. `LED_CENTER` remains available for the website's animated
light, and `M_SHELL` identifies the body material for the finish picker.

This is an exterior concept with illustrative chambers, not an internal
engineering or fabrication model. The complete nine-part internal exploded
view is not implemented in this asset. `SHELL_FRONT` currently names the whole
outer body. Website integration will need to account for those limitations
and the new pod location; replacing the production file alone is insufficient.

## Rebuild

Run Blender in background mode with `--python build_model.py` to regenerate
the GLBs, Blender scene and four renders. Add `-- --draft` to render only the
first image. The status file carries a build revision so the live preview
loads changed geometry without retaining an older export.
