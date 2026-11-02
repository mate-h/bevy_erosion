# Bevy Erosion

Real-time GPU-accelerated hydraulic erosion simulation built with Bevy.

<img width="976" height="591" alt="Frame" src="https://github.com/user-attachments/assets/cbdb4531-e8a0-4593-ac26-2c54796bab51" />

Based on the Smooth Fluvial Erosion algorithm from Kruger Terrain Tools Houdini Asset.

https://samk9632.gumroad.com/l/KTTforHoudini

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
- **Water brush**: rain erosion trails onto the terrain under the cursor, or paint seas that erosion drains into

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
| `7` | Curvature map preview |
| `8` | Age map preview |
| `B` | Toggle the water brush (left drag rains erosion trails, right drag orbits) |
| `V` | Switch the brush between rain and sea (Shift + drag erases sea) |
| `[` / `]` | Shrink / grow the water brush |
| `C` | Clear painted sea |

## Running

```bash
cargo run --release
```

For development use the file watcher feature for hot-reloading shaders:

```bash
cargo run --features=bevy/file_watcher
```
