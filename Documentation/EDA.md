# Retina Blood Vessel Feature Analysis (EDA)

This document explains the exploratory data analysis (EDA) performed on extracted vessel network features from the mouse
retinal vasculature segmentation pipeline.

## Project Context

This EDA supports the study of **mouse retinal vasculature development** from postnatal day P2 to P7. During this
period, two critical processes occur:

1. **Angiogenesis** - New blood vessels grow from existing ones by sprouting
2. **Vascular Remodeling** - Structural alterations including vessel regression, diameter changes, and artery-vein
   specification

The vessel network spreads radially from the optic nerve head across the retina during P0-P7.

## Data Structure

### Input Files (per image patch)

| File Pattern                 | Description                                          |
|------------------------------|------------------------------------------------------|
| `{filename}_alldata.xlsx`    | Vessel segment features (length, width, tortuosity)  |
| `{filename}_degreedata.xlsx` | Network node features (degree, distance from center) |
| `{filename}_network.png`     | Network visualization image                          |

### Feature Descriptions

#### Vessel Segment Features (`alldata.xlsx`)

| Feature          | Description                                                          |
|------------------|----------------------------------------------------------------------|
| `nodespair`      | Pair of nodes defining the vessel segment endpoints                  |
| `node1`, `node2` | Coordinates of segment endpoints                                     |
| `line`           | Straight-line (Euclidean) distance between endpoints                 |
| `length`         | Actual vessel length along the curve (pixels)                        |
| `width`          | Average vessel width (pixels)                                        |
| `width_var`      | Variance in vessel width along the segment                           |
| `tortuosity`     | Ratio of actual length to straight-line distance (curviness measure) |
| `curve`          | Pixel coordinates along the vessel path                              |

#### Network Node Features (`degreedata.xlsx`)

| Feature    | Description                                                            |
|------------|------------------------------------------------------------------------|
| `nodes`    | Node coordinates (x, y)                                                |
| `distance` | Distance from image center (relates to radial growth from optic nerve) |
| `degree`   | Number of vessel connections at this node                              |

**Node Degree Interpretation:**

- **Degree 1**: Endpoints (vessel tips, sprouting fronts)
- **Degree 2**: Continuation points (vessel passes through)
- **Degree 3**: Bifurcations (vessel splits into two)
- **Degree 4+**: Higher-order junctions (trifurcations, crossings)

## EDA Functions

### Individual File Analysis

#### `run_full_eda(base_path)`

Runs complete EDA for a single feature file. Includes:

1. **Network Visualization** - Display the extracted vessel network image
2. **Statistical Summary** - Descriptive statistics for all features
3. **Morphology Distributions** - Histograms and box plots for length, width, tortuosity
4. **Node Degree Analysis** - Branching pattern analysis (bifurcations, endpoints)
5. **Feature Correlations** - Correlation heatmap and scatter plots
6. **Tortuosity Analysis** - Categorization (Very Low, Low, Moderate, Normal, High)
7. **Outlier Detection** - IQR-based outlier identification
8. **Summary** - Key findings and statistics

**Usage:**

```python
run_full_eda(available_files[0])
# or
run_full_eda('Data/outputs/features/p4-from 5-5-2 M_patch_0')
```

#### Individual Analysis Functions

| Function                                          | Purpose                                          |
|---------------------------------------------------|--------------------------------------------------|
| `load_feature_data(base_path)`                    | Load alldata, degreedata, and network image path |
| `display_network_image(network_path)`             | Display vessel network visualization             |
| `show_statistical_summary(segments_df, nodes_df)` | Print descriptive statistics                     |
| `plot_morphology_distributions(segments_df)`      | Histograms for length, width, tortuosity         |
| `plot_node_degree_analysis(nodes_df)`             | Branching pattern bar charts and pie charts      |
| `plot_feature_correlations(segments_df)`          | Correlation heatmap and scatter plots            |
| `analyze_tortuosity(segments_df)`                 | Tortuosity categorization and analysis           |
| `detect_and_plot_outliers(segments_df)`           | IQR-based outlier detection                      |

### Developmental Analysis (P2-P7)

#### `run_developmental_analysis(file_paths)`

Compares vessel features across postnatal days P2-P7 to study angiogenesis and vascular remodeling.

**Analysis Includes:**

1. **Summary Statistics Table** - Metrics for each postnatal day
2. **Vessel Morphology Development**
    - Box plots: Length, width, tortuosity by day
    - Trend lines: Mean values with standard deviation
3. **Branching Pattern Development**
    - Total network nodes per day
    - Bifurcation ratio trend (vessel branching)
    - Endpoint ratio trend (vessel tips/sprouting)
    - Vessel spread from optic nerve (distance from center)
4. **Feature Distribution Comparison** - Violin plots
5. **Key Developmental Findings** - Percentage changes from P2 to P7

**Usage:**

```python
segments_combined, nodes_combined = run_developmental_analysis(available_files)
```

#### Developmental Analysis Functions

| Function                                                  | Purpose                                   |
|-----------------------------------------------------------|-------------------------------------------|
| `extract_postnatal_day(filename)`                         | Parse postnatal day (P2-P7) from filename |
| `load_all_data_by_day(file_paths)`                        | Load and combine all data with day labels |
| `plot_developmental_morphology(segments_df)`              | Morphology trends across P2-P7            |
| `plot_developmental_branching(nodes_df)`                  | Branching trends across P2-P7             |
| `plot_developmental_summary_table(segments_df, nodes_df)` | Summary table by day                      |
| `plot_violin_comparison(segments_df)`                     | Violin plots comparing distributions      |

## Biological Interpretation

### Vessel Morphology Metrics

- **Length**: Longer vessels indicate more mature vascular network
- **Width**: Vessel diameter changes during remodeling (arteries vs veins)
- **Tortuosity**: High tortuosity may indicate pathological conditions or active remodeling

### Network Metrics

- **Bifurcation Ratio**: Higher ratios indicate more complex branching (mature network)
- **Endpoint Ratio**: Higher ratios indicate more sprouting fronts (active angiogenesis)
- **Distance from Center**: Increasing distance indicates radial expansion from optic nerve

### Expected Developmental Trends (P2-P7)

| Metric               | Expected Trend          | Biological Meaning                   |
|----------------------|-------------------------|--------------------------------------|
| Vessel Length        | Increase                | Network maturation                   |
| Vessel Width         | Variable                | Artery-vein differentiation          |
| Tortuosity           | Decrease                | Network optimization                 |
| Bifurcation Ratio    | Increase                | Complex branching                    |
| Endpoint Ratio       | Decrease then stabilize | Reduced sprouting as network matures |
| Distance from Center | Increase                | Radial expansion                     |

## Quick Start

```python
# 1. List available files
available_files = list_available_files()

# 2. Run EDA on a single file
run_full_eda(available_files[0])

# 3. Run developmental analysis (P2-P7 comparison)
segments_combined, nodes_combined = run_developmental_analysis(available_files)

# 4. Use individual functions for custom analysis
segments_df, nodes_df, network_path = load_feature_data(available_files[0])
plot_morphology_distributions(segments_df)
plot_node_degree_analysis(nodes_df)
```

## File Naming Convention

Files follow the pattern: `p{day}-{sample}_patch_{n}`

- `p{day}`: Postnatal day (p2, p3, p4, p5, p6, p7)
- `{sample}`: Sample identifier (e.g., "from 5-5-2 M", "x 001 M")
- `patch_{n}`: Patch number from the original image

Example: `p4-from 5-5-2 M_patch_0` = Postnatal day 4, sample "from 5-5-2 M", patch 0
