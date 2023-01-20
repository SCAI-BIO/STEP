# STEP
Data and code for STEP (Siamese Twin deep sequence Embedding of Proteins) approach.

## Relevant files
1. ''ppi_finetuning.py'' tackles the finetuning and prediction for use case 1 (brain tissue-specific ppi prediction)
2. ''ppi_virhostnet_finetuning.py'' tackles the finetuning and prediction for use case 2 (sars-cov-2 spike ppi predisction)

## Prepare Conda Environment
1. Create the appropriate Conda env 
```
cd ~/git/STEP
conda env create -n STEP -f environment.yml
```
2. For GPU processing, install the correct pytorch package (see environment.yml for version and check out https://pytorch.org/get-started/previous-versions/ for specific commands)
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Activate Conda env
```
conda activate STEP
```

## Training a model
1. ''ppi_finetuning.py'': Use VSCode and run through the "Run and Debug" option.
```
cd ~/git/STEP
conda activate STEP

# set correct PYtHONPATH from the .env file, which is also used by VSCode
set -o allexport && source .env && set +o allexport  

# Print out all arguments
python src/ppi_finetuning.py --help

# Train a model
python src/ppi_finetuning.py --accelerator gpu --devices 2 --num_sanity_val_steps 0
```

## Manuscript to cite 
Madan, Sumit, Victoria Demina, Marcus Stapf, Oliver Ernst, and Holger Fröhlich. 2022. “Accurate Prediction of Virus-Host Protein-Protein Interactions via a Siamese Neural Network Using Deep Protein Sequence Embeddings.” Patterns 3 (9): 100551. https://doi.org/10.1016/j.patter.2022.100551.

