# Dogs vs Cats Classification Assignment

This project implements an end-to-end Convolutional Neural Network (CNN)
workflow using a trimmed **5,000‑image Dogs vs Cats dataset**.\
The notebook follows the required structure for data preparation,
exploratory analysis, model building, evaluation, and interpretation.

------------------------------------------------------------------------

## 📁 Project Structure

    ├── data/                         # Trimmed dataset (5000 images)
    ├── notebook.ipynb               # Full solution notebook
    ├── models/                      # Saved best model weights
    ├── README.md                    # Project documentation

------------------------------------------------------------------------

## ✅ 1. Data Acquisition & Preprocessing

-   Dataset is loaded using
    `tf.keras.utils.image_dataset_from_directory`.
-   Randomly sampled to **5,000 images** as required.
-   No manual separation of cats/dogs --- sampling is done directly from
    the original dataset.
-   Dataset is split into **train** and **validation** sets.
-   Applied preprocessing:
    -   Resizing
    -   Rescaling
    -   Caching + Prefetching for performance

------------------------------------------------------------------------

## 📊 2. Exploratory Data Analysis (EDA)

EDA includes: - Visual inspection of sample images\
- Class distribution plots\
- Image size/shape checks\
- Basic statistics\
- Understanding potential challenges (lighting variation, orientation,
background differences)

------------------------------------------------------------------------

## 🧠 3. Model 1 --- Vanilla CNN

-   A custom Convolutional Neural Network built from scratch.

-   Trained using:

    ``` python
    history_vanilla = vanilla_cnn.fit(
        train_ds_prep,
        validation_data=val_ds_prep,
        epochs=30,
        callbacks=vanilla_callbacks
    )
    ```

-   Uses callbacks:

    -   **ModelCheckpoint** to save the best model
    -   **EarlyStopping** to stop when validation does not improve

------------------------------------------------------------------------

## 🧠 4. Model 2 --- Fine‑Tuned VGG16

-   Loaded VGG16 pretrained on ImageNet
-   Frozen base layers → trained top layers
-   Later unfrozen top convolution blocks for fine‑tuning
-   Includes learning‑rate scheduling and model checkpointing
-   Proper validation monitoring to avoid overfitting

------------------------------------------------------------------------

## 📈 5. Model Evaluation & Comparison

Metrics included: - Accuracy - Precision, Recall, F1‑Score - Confusion
Matrix - Precision‑Recall Curve

Both models' **best saved versions** are loaded before analysis.

### Additional analysis:

-   Visualization of:
    -   Training curves (accuracy & loss)
    -   Confusion matrices
    -   PR curves
-   Identifying misclassified examples and explaining reasons.

------------------------------------------------------------------------

## 🧪 6. Error Analysis

For each model: - Extracted misclassified images - Explained possible
reasons: - Poor lighting\
- Unusual angles\
- Animals partially occluded\
- Confusing features (dogs resembling cats)

------------------------------------------------------------------------

## 📝 7. Conclusions

-   Summary comparing Vanilla CNN vs VGG16\
-   Which model performs better\
-   Why transfer learning provides stronger results\
-   Recommendations for future improvements:
    -   Data augmentation\
    -   More epochs\
    -   Additional fine‑tuning\
    -   Regularization (Dropout, BatchNorm)

------------------------------------------------------------------------

## 📚 References

-   Goodfellow, Bengio & Courville (2016) --- *Deep Learning*
-   O'Shea & Nash (2015) --- CNN introduction
-   Prechelt (1998) --- Early stopping
-   TensorFlow Documentation

------------------------------------------------------------------------

## 👩🏽‍💻 Author

**Lesley Wanjiku Kamamo**\
Business Analytics & AI/ML Student\
Conestoga College
