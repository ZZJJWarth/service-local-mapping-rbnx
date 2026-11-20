# Semantic model

The runtime service accepts an optional ONNX segmentation model:

```yaml
semantic_model_path: models/minecraft_outdoor_seg.onnx
```

Relative paths resolve against this package's root. Deployment config is the
only way to set the path.

## Contract

- Input: `float32[1, 3, H, W]`, RGB, ImageNet mean/std normalization (see
  `sensors.normalize_rgb`), NCHW.
- Output: `float32[1, C, h, w]` logits.
- `C` must be 7 and the class order is fixed:

  | index | class | cost | hazard |
  |---|---|---|---|
  | 0 | path | 0 | no |
  | 1 | grass | 15 | no |
  | 2 | vegetation | 100 | yes |
  | 3 | water | 100 | yes |
  | 4 | lava | 100 | yes |
  | 5 | structure | 100 | yes |
  | 6 | unknown | 20 | no |

- `H`/`W` come from `semantic_width` / `semantic_height` (default 320x192).
- CPU inference only (`CPUExecutionProvider`, one thread).

## Safety

The model is advisory. Geometry (lidar/depth) is integrated before semantics
and always wins. Missing file, missing onnxruntime, malformed output, or a
single inference exception disables semantics for the process and leaves the
geometric map running.

## Training data

Collect outdoor RGB in the target world, label pixels offline, and export the
model. Do not train with runtime block IDs or world coordinates: the service
never receives them. Record the model SHA-256 in the deployment manifest or
release notes, for example:

```sh
sha256sum models/minecraft_outdoor_seg.onnx
```