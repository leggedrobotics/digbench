# terra_benchmark
Benchmarks for the [Terra](https://github.com/leggedrobotics/Terra) environment.

## Benchmarks
All the benchmarks in the repo are produced starting from the
[OpenStreetMap](https://www.openstreetmap.org/) data, using the [OSMnx plugin](https://osmnx.readthedocs.io/en/stable/).
They all include buildings from the Zurich area.

There are 4 types of benchmarks:
1. crop
    - a crop of the map as-is, converted to Terra format
2. inverse crop
    - the inverse of the same crop images (e.g. the street is to dig, the building is not)
3. building mix
    - a random ensemble of individually-cropped buildings in the same image
4. inverse building mix
    - the inverse of the same building mix images

The terrains in the benchmark are discretized with a 1m tile size, and include maps ranging from 20 to 60 meters of edge size.


## Measures
The benchmark script outputs measures of:
- **success_rate**: #completed_environments / #total_environments
- **coverage**: #digged_tiles / #total_tiles_to_dig
- **avg_path_length**: average path traversed over the environments
- **path_lengths**: path traversed per environment
- **avg_workspaces**: total number of dig actions
- **workspaces**: number of dig actions per environment
- **avg_volume_overhead**: average volume_moved / volume_to_move
- **volume_overhead**: volume_moved / volume_to_move per environment


## Usage
Run the evaluation script selecting at least the model you'd like to evaluate, the benchmark name, and the output folder.
~~~
python eval.py --model your_model --benchmark benchmark_name
~~~

The script will produce:
- plots for the benchmark measures
- .pkl files containing data useful for further data analysis

## Visualization
If you wish to visualize a specific episode of the benchmark, you can run:
~~~
python visualize.py --model your_model --benchmark benchmark_name --episode episode_number
~~~

## Baselines
TODO
