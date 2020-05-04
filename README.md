# VGGNet for modified MNIST dataset
Repository for deep learning image recognition (on a modified MNIST dataset)

## Final Setup
> The below setup instructions are outdated, and only run a custom CNN which achieves low (27%) accuracy on the training set for unknown reasons.
> You only need the `FINAL_VGG.ipynb` file - we run it in Google Colab with their GPUs. This final model achieved a ~95% accuracy on the modified MNIST dataset during training, and 96% on the held-out Kaggle test set.

## Setup
1. Clone this repository
2. Start a virtual environment 
```bash
python3 -m venv env
source env/bin/activate
```
3. Install project dependencies
```
pip install -r requirements.txt
```
4. Run the driver file
