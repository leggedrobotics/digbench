# Digbench
Maps for the [Terra](https://github.com/leggedrobotics/Terra) environment.

## Maps
There are currently 2 types of "main" map types:
1. Foundations
    - we take building foundations from the [OpenStreetMap](https://www.openstreetmap.org/) API, and we project it onto the grid map
2. Trenches
    - procedurally-generated trenches with 1, 2, or 3 axes, obstacles, no-dumping zones and terminal dumping constraints

There are also other scripts that generate simpler maps, used for the propedeutical phase of learning of the more difficult maps.
These are generated by the following scripts:
- `generate_rectangles.py` 
- `procedural_squares.py` 
- `procedural_onetile.py` 

## How to generate maps
For the 2 main types of maps, we generate them as RGB maps using the `generate_datasets.py` script (the script is made to generate all the map types at the same time, so you need to comment out what you don't need). On top of the RGB, also occupancies and metadata are generated.
To convert the output to Terra-compatible format, use the script `postprocessing.py` and check the results with the script `visualize_npy.py`.
