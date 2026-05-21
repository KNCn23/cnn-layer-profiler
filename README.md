# CNN Layer Profiler

A lightweight C implementation of common CNN layers with per-layer timing, memory, and FLOP profiling. Includes a Python script for visualizing results.

## Architecture

```
├── include/
│   ├── tensor.h      # Tensor struct and operations
│   ├── layers.h      # Layer forward-pass declarations
│   └── profiler.h    # Profiling structs and utilities
├── src/
│   ├── tensor.c      # Tensor memory management
│   ├── layers.c      # Conv2D, ReLU, MaxPool, Flatten, FC
│   ├── profiler.c    # Timing, CSV export, table printer
│   └── main.c        # Demo network (VGG-style, 224×224 input)
├── python/
│   ├── visualize.py  # Bar charts via matplotlib (ASCII fallback)
│   └── requirements.txt
└── Makefile
```

## Build & Run

```bash
make run          # build + execute the demo network
```

Sample output:

```
Layer                   Time(ms)    Memory(KB)          FLOPs
---------------------   --------    ----------          -----
Conv2D-1  (3→64,3×3)      42.318      2450.00     173,408,256
ReLU-1                     0.823      2450.00               0
MaxPool-1 (2×2,s2)         0.412       612.50               0
...
TOTAL                     89.201     ...           ...
```

## Visualize

```bash
pip install -r python/requirements.txt
python python/visualize.py profile_results.csv
```

Generates a PNG with three bar charts: time, memory, and FLOPs per layer.

## Requirements

- GCC (C11) with `-lm`
- Python 3.8+ (optional, for plots)
- macOS / Linux

## License

MIT
