# Bevy Erosion

Real-time GPU-accelerated hydraulic erosion simulation built with Bevy.

## Features

- **Particle-based erosion simulation** running entirely on GPU compute shaders
- **Multiple visualization modes**:
  - PBR shading with atmospheric effects
  - Flow map (direction & magnitude)
  - Sediment deposition heat map
  - Erosion intensity visualization
  - Height contours
  - View-space normals
- **Interactive controls** for real-time simulation

## Controls

| Key | Action |
|-----|--------|
| `R` | Reset simulation |
| `Space` | Pause/Resume |
| `E` | Step one iteration (when paused) |
| `1` | PBR shading mode (default) |
| `2` | Flow map preview |
| `3` | Sediment mask preview |
| `4` | Erosion mask preview |
| `5` | Height map preview |
| `6` | View-space normals preview |

## Running

```bash
cargo run --release
```

For development use the file watcher feature for hot-reloading shaders:

```bash
cargo run --features=bevy/file_watcher
```
