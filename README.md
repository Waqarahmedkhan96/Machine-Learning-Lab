# Machine Learning Lab

This repository contains machine learning lab material, course presentations, and six completed assignment solutions. The assignments cover regression, supervised classification, neural networks, dimensionality reduction, clustering, and model evaluation.

## Repository Structure

| Folder | Contents |
|---|---|
| [`1 presentations`](<1 presentations>) | Lecture slides and supporting course PDFs |
| [`2 assignments`](<2 assignments>) | Six assignment folders with notebooks, datasets, reports, and evaluation images |

## Assignment Overview

| # | Assignment | Main task | Model or method | Key evaluation |
|---:|---|---|---|---|
| 1 | [Car prices](<2 assignments/1 Car prices (updated)>) | Predict electric vehicle prices and classify cars as cheap or expensive | Linear Regression, Ridge, kNN | OLS RMSE 52,673 DKK, R^2 0.864; kNN accuracy 0.943 |
| 2 | [Candidates part 1](<2 assignments/2 Candidates part 1>) | Predict Danish political party from candidate-test answers | Decision Tree, Random Forest, Gradient-Boosted Trees | Random Forest selected using accuracy and macro F1 |
| 3 | [Mushroom foraging](<2 assignments/3 Mushroom foraging>) | Classify mushrooms as edible or poisonous | Logistic Regression | Accuracy 0.8167, poisonous recall 0.8241, ROC AUC 0.8816 |
| 4 | [Detecting exoplanets](<2 assignments/4 Detecting exoplanets>) | Classify Kepler objects as candidates or false positives | Logistic Regression, SVM | Logistic Regression test accuracy 0.902, F1 0.905 |
| 5 | [Sentiment analysis](<2 assignments/5 Sentiment analysis>) | Classify movie reviews as positive or negative | Bag-of-Words, MLP neural network | Test accuracy 0.8830 |
| 6 | [Candidates part 2](<2 assignments/6 Candidates part 2>) | Explore political similarity and candidate clusters | PCA, K-Means, hierarchical clustering, DBSCAN | PCA explains about 52% with PC1 and PC2; best K-Means silhouette about 0.293 at k=2 |

## Assignment 1: EV Car Price Prediction and Classification

This assignment starts as a supervised regression problem where the target is `Price (DKK)`. The solution compares manual normal-equation linear regression, Scikit-Learn OLS regression, regularized regression, and a final kNN classification task after converting prices into cheap or expensive labels.

Key results:

- Manual linear regression: R^2 about 0.85, RMSE about 55,000 DKK.
- Scikit-Learn OLS: R^2 about 0.864, RMSE 52,673 DKK.
- kNN classifier: best model used `k=7` with Manhattan distance and reached 94.3% accuracy.

![Assignment 1 regression metrics](<2 assignments/1 Car prices (updated)/assets/car_prices_regression_metrics.png>)

![Assignment 1 kNN metrics](<2 assignments/1 Car prices (updated)/assets/car_prices_knn_metrics.png>)

## Assignment 2: Candidate Test 2022, Supervised Classification

This assignment uses Danish DR and TV2 candidate-test answer data to predict a candidate's political party. It combines descriptive analysis with supervised multiclass classification and compares Decision Tree, Random Forest, and Gradient-Boosted Trees.

Key results:

- Random Forest was selected as the best model using cross-validation.
- Accuracy is used for overall correctness.
- Macro F1 is used because party classes are imbalanced and smaller parties should count equally.
- The confusion matrix is used to inspect which parties are predicted correctly or confused.

![Assignment 2 baseline and final scores](<2 assignments/2 Candidates part 1/model_notes_assets/evaluation_baseline_final_scores.png>)

![Assignment 2 confusion matrix](<2 assignments/2 Candidates part 1/model_notes_assets/evaluation_confusion_matrix.png>)

![Assignment 2 cross-validation comparison](<2 assignments/2 Candidates part 1/model_notes_assets/evaluation_cv_comparison.png>)

## Assignment 3: Mushroom Toxicity Classification

This assignment is a supervised binary classification task. The goal is to classify mushrooms as edible or poisonous using structured mushroom features. The solution focuses on careful preprocessing, leakage-aware splitting, logistic regression, hyperparameter tuning, and safety-aware model interpretation.

Key results:

- Final model: Logistic Regression with `C = 0.1`.
- Accuracy: 0.8167.
- Precision for poisonous: 0.8422.
- Recall for poisonous: 0.8241.
- F1 score for poisonous: 0.8330.
- ROC AUC: 0.8816.

Recall for the poisonous class is the most important metric because a false negative means a poisonous mushroom is predicted as edible.

![Assignment 3 validation comparison](<2 assignments/3 Mushroom foraging/report_assets/validation_method_comparison.png>)

![Assignment 3 final metrics](<2 assignments/3 Mushroom foraging/report_assets/final_test_metrics.png>)

![Assignment 3 confusion matrix](<2 assignments/3 Mushroom foraging/report_assets/test_confusion_matrix.png>)

## Assignment 4: Kepler Exoplanet Classification

This assignment classifies Kepler Objects of Interest as exoplanet candidates or false positives. It compares Logistic Regression and SVM on scaled numerical features after cleaning and preparing the Kepler dataset.

Key results:

- Final selected model: Logistic Regression.
- Logistic Regression validation accuracy and F1: 0.994.
- SVM validation accuracy and F1: about 0.993.
- Final Logistic Regression test accuracy: 0.902.
- Final Logistic Regression test F1: 0.905.

The report notes that final test performance is the better generalization estimate because validation performance was higher than test performance.

![Assignment 4 evaluation metrics](<2 assignments/4 Detecting exoplanets/model_report_assets/evaluation_metrics_summary.png>)

![Assignment 4 confusion matrices](<2 assignments/4 Detecting exoplanets/model_report_assets/confusion_matrices_test.png>)

## Assignment 5: IMDB Sentiment Analysis

This assignment builds a supervised text classification model for IMDB movie reviews. Reviews are encoded with Bag-of-Words and classified with a one-hidden-layer neural network.

Key results:

- Training set: 16,000 reviews.
- Validation set: 4,000 reviews.
- Test set: 5,000 reviews.
- Final test accuracy: 0.8830.
- Confusion matrix: `[[2236, 264], [321, 2179]]`.

![Assignment 5 notebook output](<2 assignments/5 Sentiment analysis/assets/notebook_evaluation_output_1.png>)

![Assignment 5 evaluation metrics](<2 assignments/5 Sentiment analysis/assets/evaluation_metrics.png>)

![Assignment 5 confusion matrix](<2 assignments/5 Sentiment analysis/assets/confusion_matrix.png>)

## Assignment 6: Candidate Test 2022, Similarity Analysis

This assignment is an unsupervised learning project using Danish candidate-test answer data. Instead of predicting a target label, the goal is to explore similarity, political structure, party positioning, and candidate clusters.

Main methods:

- PCA for dimensionality reduction and political mapping.
- K-Means for cluster analysis.
- Hierarchical clustering for party similarity.
- DBSCAN for density-based clustering and noise detection.
- Euclidean distance for candidate agreement and disagreement.

Key results:

- PC1 and PC2 together explain about 52% of the answer variation.
- K-Means was tested from `k=2` to `k=14`.
- Best K-Means result: `k=2`, silhouette score about 0.293.
- Because this is unsupervised learning, there is no accuracy, precision, recall, or F1 score.

![Assignment 6 PCA explained variance](<2 assignments/6 Candidates part 2/assets/pca_explained_variance.png>)

![Assignment 6 K-Means silhouette scores](<2 assignments/6 Candidates part 2/assets/kmeans_silhouette_scores.png>)

![Assignment 6 DBSCAN sensitivity](<2 assignments/6 Candidates part 2/assets/dbscan_eps_sensitivity.png>)

## Notes

Each assignment folder includes the notebook, related datasets, a written report, and evaluation images where applicable. The reports provide exam-style explanations of the workflow, model choices, metrics, and limitations.
