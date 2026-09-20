# Wane One: 3D model spec for the website

The site loads `wane-one.glb` from the same folder as `index.html`. If the file is missing or
does not follow this spec, the page falls back to the device built in code, so nothing breaks
while the model is in progress. Add `?model=0` to the URL to force the built-in device, or
`?model=other.glb` to try another file.

`wane-one.baseline.glb` is an export of the current built-in device with every name below
already in place. Open it in Blender and replace the geometry part by part; the names carry over.

## File
- Format: glTF binary (`.glb`), one file, textures embedded. Draco compression is fine.
- Size: under 3 MB. The page is 180 KB, the model should not be twenty times that.
- Units: anything. The page scales the model so its height is 105 mm and centres it. Keep
  the proportions from the concept sheet: 105 x 22 x 12 mm.
- Orientation: +Y up, +Z is the front (the face with the light), +X is the user's right.
- Polygons: aim for under 150k triangles. Smooth shading on the shell, no baked lighting.

## The nine parts, each a top-level object (empty or mesh) whose name STARTS with its code
The page explodes, labels and lists exactly these. Missing parts are skipped with a warning.

| Object name starts with | What it is | Notes |
|---|---|---|
| `WN-01` | Mouthpiece cap | brushed aluminium taper, flat top |
| `WN-02` | Pod, the lower section | contains the window slot; put the liquid mesh inside it |
| `WN-03` | Pressure sensor | small block near the airway, high in the body |
| `WN-04` | Heater driver | small board just above the pod |
| `WN-05` | MCU + BLE board | runs the length of the body, USB-C at its foot |
| `WN-06` | Battery | middle of the body |
| `WN-07` | Haptic motor | small cylinder |
| `WN-08` | Light ring assembly | the recessed dial, lens and PCB behind it |
| `WN-09` | Shell, the upper body | split into two children, see below |

Anything not named `WN-xx` is treated as part of the shell.

## Named children the page looks for
| Name | Where | What the page does with it |
|---|---|---|
| `SHELL_FRONT` | child of `WN-09` | slides forward in the exploded view so the internals show; the rest of `WN-09` slides back |
| `LED_CENTER` or `LED_LENS` | anywhere, usually inside `WN-08` | the live light (track, arc, glow) is drawn on this point, facing +Z. An empty at the dial centre is enough; if it is a mesh named `LED_LENS`, its width sets the arc size |
| `STATUS_DOT` or `LED_DOT` | anywhere | the small teal dot; drawn on this point. If absent it goes just under the ring |
| `LIQUID_N` | inside `WN-02` | the visible liquid in the window. It is scaled from its bottom edge as the dose slider moves, so model it at full height |
| `WINDOW` | inside `WN-02` | optional, the slot bezel and glass, used to focus the camera |

Do not model the lit arc, the glow or the dot as geometry: the page draws them, animates them,
and turns them amber and dim for the ring states.

## Materials
Name the shell material `M_SHELL`. The finish picker retints it live (Pine, Sage, Bone) and sets
its roughness and metalness, so leave it a plain PBR material with no baked colour texture.
Every other material is used as authored. Suggested names: `M_CAP` (brushed metal), `M_GLASS`
(window, transparent), `M_LENS` (dark dial recess), `M_BOARD`, `M_CELL`, `M_PORT`.

## Checklist before sending
- [ ] Nine objects named `WN-01` to `WN-09` at the top level
- [ ] `SHELL_FRONT` inside `WN-09`, `LIQUID_N` inside `WN-02`, `LED_CENTER` inside `WN-08`
- [ ] `M_SHELL` on the shell and the pod shell
- [ ] +Y up, +Z front, proportions 105 x 22 x 12
- [ ] Under 3 MB, textures embedded
- [ ] Drop it next to `index.html` as `wane-one.glb`, reload, check the console says
      "authored model loaded"
