# Data Notes

This directory contains the released annotations and dataset-construction metadata for the four paper datasets:

- `agedb`
- `poster_score`
- `boneage`
- `imdb_wiki`

## Path Convention

The final train/test annotation files use relative image paths:

- `images/...`

These paths are designed to match the Hugging Face dataset layout, where each dataset subset contains:

- an `images/` directory
- the corresponding train/test annotation files

## Included Files

For each dataset, this repository keeps:

- final train/test annotations used in the paper
- split scripts or preprocessing scripts when available
- metadata csv files
- distribution plots and weighting files when they are lightweight and directly relevant

The full image assets are not stored in this GitHub repository.
