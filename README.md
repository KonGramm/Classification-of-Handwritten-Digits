# Handwritten Digit Classification via SVD Basis Representation

A Python implementation of a classic numerical-linear-algebra approach to classifying
handwritten digits from the US Postal Service database, using the Singular Value
Decomposition (SVD) of each digit class as a low-dimensional basis.

## Overview

Each digit class (0–9) is represented by a matrix of training images. For every class,
we compute the SVD of that matrix and keep the first *k* left singular vectors (5–20) as
an orthonormal basis for the class. An unknown test digit is then classified by checking,
for each class, how well the digit can be reconstructed from that class's basis — measured
via the residual of the projection (least-squares problem). The digit is assigned to the
class with the smallest residual.

This mirrors the approach described in Eldén, *Matrix Methods in Data Mining and Pattern
Recognition*, and is a well-known benchmark problem for SVD-based classification.

## Data

- **`data.xlsx`** — contains four sheets, matching the original text-file format of the
  US Postal Service digit database:
  - `azip` — training images (256-dimensional vectors, from 16×16 pixel images)
  - `dzip` — training labels
  - `testzip` — test images
  - `dtest` — test labels

Each image is a flattened 16×16 grayscale image (256 pixel values).

## Method

1. **Build class matrices.** Training images are grouped by label into 10 matrices
   `A_0, ..., A_9`, one per digit class.
2. **Compute the SVD** of each class matrix `A_i = U_i S_i V_i^T`.
3. **Classify** each test image `x` by computing, for each class `i` and a chosen number
   of basis vectors `k`:

   ```
   residual_i = || (I - U_i[:, :k] U_i[:, :k]^T) x ||
   ```

   and assigning `x` to the class with the smallest residual.
4. **Evaluate** accuracy across `k = 5, ..., 20` basis vectors to find the best trade-off
   between dimensionality and classification accuracy.

## Tasks addressed

1. **Accuracy tuning** — classification accuracy is computed for each value of `k`
   (5–20 basis vectors) and plotted, to identify the optimal number of basis vectors.
2. **Per-digit difficulty analysis** — a classification report breaks down precision/recall
   per digit, and misclassified test images are plotted to visually inspect which digits
   are hardest to classify (often poorly/ambiguously handwritten samples).
3. **Class-specific basis size** *(exploratory)* — investigates whether some digit classes
   could use fewer basis vectors than others without losing accuracy, based on the decay
   of singular values per class.

## Requirements

```
pandas
numpy
matplotlib
scipy
scikit-learn
openpyxl   # required by pandas to read .xlsx files
```

Install with:

```bash
pip install pandas numpy matplotlib scipy scikit-learn openpyxl
```

## Usage

1. Place `data.xlsx` in the same folder as the script (or update the file paths in the
   script to point to its location).
2. Run the script:

   ```bash
   python digit_classification_svd.py
   ```

3. The script will:
   - Display a sample digit image
   - Compute the SVD for each of the 10 digit classes
   - Classify all test digits for `k = 5` to `20` basis vectors
   - Plot accuracy vs. number of basis vectors
   - Print a classification report for the best-performing `k`
   - Display a grid of misclassified digits with their true and predicted labels

> **Note:** the script currently reads `data.xlsx` from a hardcoded local path — update
> the `pd.read_excel(...)` calls near the top of the script to point to your own copy of
> the data before running.

## Results

- Classification accuracy is reported as a function of the number of basis vectors used
  per class (typically peaking somewhere in the 10–15 range for this dataset).
- Misclassifications are concentrated in visually ambiguous or poorly written digits
  (e.g. 4s that resemble 9s, or 3s that resemble 8s).

## Files

| File | Description |
|---|---|
| `digit_classification_svd.py` | Main script: SVD basis construction, classification, and evaluation |
| `data.xlsx` | Training/test images and labels |
| `Project_Description.docx` | Original assignment description |

## Author

Konstantinos Grammenos
