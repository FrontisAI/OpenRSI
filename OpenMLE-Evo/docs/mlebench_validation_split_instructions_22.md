# MLE-bench 22 Tasks: Deterministic Validation Split Instructions

This file gives task-specific self-validation split instructions for the 22-task MLE-bench subset used in the AIRA-Evo / memory experiments.

Goal: make the model's internal validation score more reliable for selecting the final node. These instructions are intended to be injected into each task prompt.

Compared with the earlier version, this version removes vague language such as "prefer", "if feasible", "usually", and "if available" wherever the lite dataset structure is known. Every task now specifies:

- exact validation unit
- exact split ratio or fold protocol
- fixed random seed or deterministic split rule
- exact validation metric
- leakage constraints

Evidence used:

- The released MLE-Bench Lite task-package structure
- The fixed task inventory and corresponding release validation audit
- The released OpenMLE-Evo validation protocol

General instruction to prepend to every task:

```text
Validation split guidance:
Use the task-specific validation protocol below exactly. The validation score must use the same metric direction as the competition metric. Fit preprocessing, feature selection, vectorizers, scalers, encoders, threshold tuning, early stopping, and model selection only on the training fold, then evaluate once on the validation fold. Use random_state=42 whenever a randomized split is specified. Print this validation metric clearly as the model-selection score. Do not choose the final solution by hidden/test/sandbox score.
```

## 1. aerial-cactus-identification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has columns `id` and `has_cactus`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train["has_cactus"] over image ids. Train on the 80% training image ids only and validate on the 20% held-out image ids only. Apply augmentation only to training images; validation preprocessing must be deterministic.

Validation metric: ROC-AUC on validation `has_cactus` probabilities. Higher is better.
```

Rationale:

The real data has `train.csv(id, has_cactus)` and images under `train/`. The official metric for this task is ROC-AUC, so validation must output probabilities and compute AUC, not accuracy. A fixed 80/20 stratified split reduces fluctuation in the positive/negative ratio and is more stable than a range like "10-20%".

## 2. aptos2019-blindness-detection

Recommended prompt instruction:

```text
Validation split protocol: train.csv has columns `id_code` and ordinal class `diagnosis`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train["diagnosis"] over image ids. Train on the 80% training image ids only and validate on the 20% held-out image ids only. Apply augmentation only to training images. If the model outputs a continuous severity score, choose the four class thresholds using only the validation fold and report the resulting validation score.

Validation metric: quadratic weighted kappa between validation `diagnosis` labels and predicted integer classes 0-4. Higher is better.
```

Rationale:

The real data is `train.csv(id_code, diagnosis)` and the official metric is QWK. When no metric was enforced, models might select nodes by accuracy, MSE, or loss, making the validation score inconsistent with the leaderboard. This version explicitly stratifies by `diagnosis` and fixes QWK as the only selection metric.

## 3. denoising-dirty-documents

Recommended prompt instruction:

```text
Validation split protocol: public data contains paired dirty images in `train/` and clean targets in `train_cleaned/` with matching png filenames. For each png filename, compute `int(hashlib.md5(filename.encode()).hexdigest(), 16) % 5`; use images with value 0 as validation and all remaining images as training. If extracting patches or tiles, extract validation patches only from validation image ids and training patches only from training image ids. Never mix patches from the same original image across train and validation.

Validation metric: RMSE over pixel values on the held-out validation images reconstructed at image level. Lower is better.
```

Rationale:

The real data has no CSV labels; it is paired pngs in `train/` and `train_cleaned/`. The biggest risk is patch leakage. Using a filename md5 hash modulo for an image-level validation set is fully deterministic and less sensitive to filename ordering or data-source ordering than "every 5th image after sorting".

## 4. detecting-insults-in-social-commentary

Recommended prompt instruction:

```text
Validation split protocol: train.csv has columns `Insult`, `Date`, and `Comment`. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by `Insult`. Fit text preprocessing and vectorizers separately inside each fold. Use the mean out-of-fold validation metric for model selection.

Validation metric: ROC-AUC on out-of-fold `Insult` probabilities. Higher is better.
```

Rationale:

The real data is binary text classification and the official metric is AUC. The dataset is small, so a single holdout shows visible seed variance; a fixed 5-fold StratifiedKFold is more stable than a random 80/20 split, and it keeps TF-IDF/vectorizers from leaking into the validation text.

## 5. dog-breed-identification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has columns `id` and `breed`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train["breed"] over image ids. Train on the 80% training image ids only and validate on the 20% held-out image ids only. Apply augmentation only to training images. Use the class order from sample_submission.csv columns after `id` when building validation probability arrays.

Validation metric: multiclass log loss over all dog breed classes in sample_submission order. Lower is better.
```

Rationale:

The real data has 120 breed probability columns and the official metric is multiclass log loss. Fixing the class order explicitly avoids incomparable log loss values caused by LabelEncoder ordering or classes missing from the validation fold.

## 6. dogs-vs-cats-redux-kernels-edition

Recommended prompt instruction:

```text
Validation split protocol: parse the binary label from train image filenames: `cat.*.jpg` is 0 and `dog.*.jpg` is 1. Use train_test_split with test_size=0.2, random_state=42, and stratify by this parsed label over image filenames. Train on the 80% training images only and validate on the 20% held-out images only. Apply augmentation only to training images; validation preprocessing must be deterministic.

Validation metric: binary log loss on validation dog probabilities. Lower is better.
```

Rationale:

The real data has no train.csv; labels live in the `cat.*.jpg` / `dog.*.jpg` filenames. The official metric is log loss, not accuracy. A fixed 80/20 stratified split prevents an overconfident model from being selected when its accuracy is high but its log loss is poor.

## 7. histopathologic-cancer-detection

Recommended prompt instruction:

```text
Validation split protocol: train_labels.csv has columns `id` and binary `label`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train_labels["label"] over image ids. Train on the 80% training image ids only and validate on the 20% held-out image ids only. Apply augmentation only to training images.

Validation metric: ROC-AUC on validation cancer `label` probabilities. Higher is better.
```

Rationale:

The real data has `train_labels.csv(id, label)` and `.tif` patch images, and the official metric is AUC. The lite data has no explicit patient/slide id, so instead of saying "if a patient id exists", this version pins a stratified image-id split directly.

## 8. jigsaw-toxic-comment-classification-challenge

Recommended prompt instruction:

```text
Validation split protocol: train.csv has labels `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. Create a deterministic stratification key:
`any_toxic = max(all six labels)`;
`label_count_bucket = min(sum(all six labels), 2)`;
`rare_any = max(severe_toxic, threat, identity_hate)`.
Use train_test_split with test_size=0.2, random_state=42, and stratify by the string key `any_toxic + "_" + label_count_bucket + "_" + rare_any`. Fit tokenizers/vectorizers only on the training fold.

Validation metric: macro mean ROC-AUC across the six toxicity labels. Higher is better.
```

Rationale:

The real data is 6-label multilabel text and the official metric is macro mean ROC-AUC. The earlier "iterative multilabel stratification" could vary by implementation when the library was unavailable; this version switches to a fixed composite stratification key that every node can execute identically.

## 9. leaf-classification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has target `species` and numeric features `margin1..64`, `shape1..64`, and `texture1..64`. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by `species`. Fit scalers, PCA, feature selection, and model early stopping separately inside each fold. Use the class order from sample_submission.csv columns after `id`.

Validation metric: mean multiclass log loss across the 5 validation folds, using the complete sample_submission class order. Lower is better.
```

Rationale:

The real data is a small 99-class tabular/image-feature task and the official metric is multiclass log loss. A fixed 5-fold suits small data better than a single holdout, and with the class order pinned, validation scores are comparable across nodes.

## 10. mlsp-2013-birds

Recommended prompt instruction:

```text
Validation split protocol: use the provided `essential_data/CVfolds_2.txt` and `essential_data/rec_labels_test_hidden.txt`. Treat records with `?` in `rec_labels_test_hidden.txt` as test/hidden records and exclude them from validation. For all known records, define the binary target as 1 if the label field is non-empty and 0 if the label field is empty. Train on known records with fold == 0 and validate on known records with fold == 1. If creating spectrograms, clips, frames, or segment features, create them after this recording-level split.

Validation metric: ROC-AUC on the validation binary bird-present probabilities. Higher is better.
```

Rationale:

The real data ships with `CVfolds_2.txt`, `rec_id2filename.txt`, and `rec_labels_test_hidden.txt`, and the sample_submission has a single `Probability` column. So this uses the official folds directly and defines a non-empty known label field as bird-present=1 and an empty label field as 0, which is clearer than "same target definition".

## 11. new-york-city-taxi-fare-prediction

Recommended prompt instruction:

```text
Validation split protocol: train.csv has target `fare_amount`. First remove only these invalid training rows: `fare_amount <= 0`, `passenger_count < 1`, `passenger_count > 6`, pickup/dropoff longitude outside [-75, -72], or pickup/dropoff latitude outside [40, 42]. Do not apply any other target-dependent row filtering before the split. On the cleaned training rows, create `fare_bin = pandas.qcut(fare_amount, q=10, duplicates="drop")`. Use train_test_split with test_size=0.2, random_state=42, and stratify by `fare_bin`. Feature engineering such as haversine distance and pickup datetime features must be computed without using validation targets for training decisions.

Validation metric: RMSE on held-out `fare_amount`. Lower is better.
```

Rationale:

The real columns are `fare_amount`, pickup/dropoff coordinates, `pickup_datetime`, and `passenger_count`, and the official metric is RMSE. The old wording "pickup year/month or date bucket, passenger-count bucket, and approximate distance bucket if feasible" was too vague; this version fixes the cleaning bounds and fare-decile stratification, keeping the long-tail fare distribution stable with a simple, consistent implementation.

## 12. nomad2018-predict-transparent-conductors

Recommended prompt instruction:

```text
Validation split protocol: train.csv has two regression targets: `formation_energy_ev_natom` and `bandgap_energy_ev`. Create `formation_bin = pandas.qcut(formation_energy_ev_natom, q=5, duplicates="drop")` and `bandgap_bin = pandas.qcut(bandgap_energy_ev, q=5, duplicates="drop")`. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by the combined string key `formation_bin + "_" + bandgap_bin`. Fit all feature engineering and model selection inside each fold.

Validation metric: mean RMSLE across `formation_energy_ev_natom` and `bandgap_energy_ev`, with predictions clipped to nonnegative values before RMSLE. Lower is better.
```

Rationale:

The real data has two targets and the official metric is the mean RMSLE over both. Nomad's val/test mismatch is noticeable, and a single random holdout is easily misleading; a fixed 5-fold target-quantile stratification keeps both target distributions stable.

## 13. plant-pathology-2020-fgvc7

Recommended prompt instruction:

```text
Validation split protocol: train.csv has one-hot disease columns `healthy`, `multiple_diseases`, `rust`, and `scab`. Create `label_name = idxmax([healthy, multiple_diseases, rust, scab])`. Use train_test_split with test_size=0.2, random_state=42, and stratify by `label_name` over image ids. Apply augmentation only to training images.

Validation metric: macro mean ROC-AUC across the four disease probability columns. Higher is better.
```

Rationale:

The real data has four one-hot disease label columns and the official metric is mean column-wise ROC-AUC. Fixing a label-combination stratification is clearer than "80/20 or 90/10" and reduces AUC fluctuation caused by small classes.

## 14. random-acts-of-pizza

Recommended prompt instruction:

```text
Validation split protocol: train.json has binary target `requester_received_pizza`. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by `requester_received_pizza`. Use only fields that also exist in test.json, plus request-time fields; do not use retrieval-only fields that are absent from test.json. Fit text vectorizers, encoders, and feature selection inside each fold.

Validation metric: mean ROC-AUC across the 5 folds on `requester_received_pizza` probabilities. Higher is better.
```

Rationale:

The real train.json contains fields absent from test.json, such as retrieval-time vote/comment fields. Beyond split instability, such fields directly cause test failures or leakage-inflated scores. This version fixes 5-fold AUC and restricts features to request-time fields that also exist in test.json.

## 15. ranzcr-clip-catheter-line-classification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has `StudyInstanceUID`, multilabel catheter columns, and `PatientID`. Use GroupShuffleSplit with n_splits=1, test_size=0.2, random_state=13, groups=train["PatientID"]. Train on the 80% patient groups only and validate on the 20% held-out patient groups only. Use only the label columns that appear in sample_submission.csv after `StudyInstanceUID` when computing the validation metric.

Validation metric: macro mean ROC-AUC across the sample_submission label columns. Higher is better.
```

Rationale:

The real data explicitly includes `PatientID`, so there is no need for "if a patient id exists". Medical X-ray tasks must split by patient to keep correlated images of the same patient from crossing train/val. `random_state=13` is pinned because, on the lite data, it leaves more validation positives for the rare label `ETT - Abnormal` than 42 does, making the AUC more stable. The sample_submission contains only the 9 predicted labels, so the metric should be computed over those submission columns.

## 16. siim-isic-melanoma-classification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has `image_name`, `patient_id`, metadata columns, and binary target `target`. Use GroupShuffleSplit with n_splits=1, test_size=0.2, random_state=42, groups=train["patient_id"]. Train on the 80% patient groups only and validate on the 20% held-out patient groups only. Fit metadata encoders and image normalization choices only on the training fold.

Validation metric: ROC-AUC on validation melanoma `target` probabilities. Higher is better.
```

Rationale:

The real data explicitly includes `patient_id` and the official metric is AUROC. Pinning a patient-level split prevents same-patient information from leaking into the validation set, which matters more than plain target stratification.

## 17. spooky-author-identification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has target `author` with classes EAP, HPL, and MWS. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by `author`. Fit TF-IDF/tokenizers, n-gram vocabulary, SVD, and calibration inside each fold. Use class order from sample_submission.csv columns after `id`.

Validation metric: mean multiclass log loss across the 5 folds in sample_submission author order. Lower is better.
```

Rationale:

The real data is three-class text and the official metric is multiclass log loss. Fixing 5-fold CV and the submission class order avoids tokenizer leakage and label-order mismatches.

## 18. tabular-playground-series-dec-2021

Recommended prompt instruction:

```text
Validation split protocol: train.csv has target `Cover_Type`. Use StratifiedShuffleSplit with n_splits=1, test_size=0.1, random_state=42, stratified by `Cover_Type`. Fit all preprocessing and model selection on the 90% training fold only, then evaluate once on the 10% validation fold.

Validation metric: classification accuracy on held-out `Cover_Type`. Higher is better.
```

Rationale:

The real data is a large-scale multiclass tabular task and the official metric is accuracy. A fixed 90/10 stratified split is stable enough while cheaper than KFold, which suits a 12h agent search.

## 19. tabular-playground-series-may-2022

Recommended prompt instruction:

```text
Validation split protocol: train.csv has binary target `target`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train["target"]. Fit all preprocessing, `f_27` feature extraction, encoders, and model selection on the training fold only.

Validation metric: ROC-AUC on validation `target` probabilities. Higher is better.
```

Rationale:

The real columns include `f_27`, but the old wording "optionally include f_27 pattern buckets" let different nodes adopt different splits. This version stratifies by the target only; `f_27` may serve as a model feature but must not change the validation-set definition.

## 20. text-normalization-challenge-english-language

Recommended prompt instruction:

```text
Validation split protocol: en_train.csv has columns `sentence_id`, `token_id`, `class`, `before`, and `after`. Convert `sentence_id` to integer. Use sentences where `sentence_id % 20 == 0` as validation, and all other sentence ids as training. Never split by token rows. Build dictionaries, rules, frequency tables, and models only from training sentences.

Validation metric: exact token-level accuracy of predicted `after` strings on validation tokens. Higher is better.
```

Rationale:

The real data has `sentence_id/token_id/class/before/after`. Pinning `sentence_id % 20 == 0` yields an ~5% sentence-level validation set that is fully deterministic and avoids row-level token leakage.

## 21. text-normalization-challenge-russian-language

Recommended prompt instruction:

```text
Validation split protocol: ru_train.csv has columns `sentence_id`, `token_id`, `class`, `before`, and `after`. Convert `sentence_id` to integer. Use sentences where `sentence_id % 20 == 0` as validation, and all other sentence ids as training. Never split by token rows. Build dictionaries, transliteration rules, frequency tables, and models only from training sentences.

Validation metric: exact token-level accuracy of predicted `after` strings on validation tokens. Higher is better.
```

Rationale:

Russian text normalization mirrors the English task, and the real data likewise has `sentence_id/token_id/class/before/after`. A fixed sentence-level modulo split prevents context leakage within the same sentence and keeps the validation set identical across attempts.

## 22. the-icml-2013-whale-challenge-right-whale-redux

Recommended prompt instruction:

```text
Validation split protocol: training audio labels are encoded in train filenames ending with `_0.aif` or `_1.aif`; parse this suffix as the binary target. Use train_test_split with test_size=0.2, random_state=42, and stratify by the parsed target over original audio clip filenames. If creating spectrograms, windows, crops, or audio features, create them after this clip-level split and never mix windows from the same clip across train and validation.

Validation metric: ROC-AUC on validation right-whale probabilities. Higher is better.
```

Rationale:

Real train filenames look like `..._TRAIN0_0.aif`, with the label in the filename suffix. The official metric is AUC. A fixed clip-level stratified split keeps windows derived from the same audio clip from crossing train/val.

## Compact Per-Task Prompt Map

If the system needs a shorter instruction block, use the following one-liners:

| Task | Validation split one-liner |
|---|---|
| aerial-cactus-identification | 80/20 image-id split stratified by `has_cactus`, seed 42; validate ROC-AUC. |
| aptos2019-blindness-detection | 80/20 image-id split stratified by `diagnosis`, seed 42; validate quadratic weighted kappa. |
| denoising-dirty-documents | Filename md5 modulo split: hash % 5 == 0 dirty/clean pairs as validation before patching; validate image-level RMSE. |
| detecting-insults-in-social-commentary | 5-fold StratifiedKFold by `Insult`, seed 42; validate mean ROC-AUC. |
| dog-breed-identification | 80/20 image-id split stratified by `breed`, seed 42; validate multiclass log loss in submission class order. |
| dogs-vs-cats-redux-kernels-edition | 80/20 image-id split stratified by filename cat/dog label, seed 42; validate binary log loss. |
| histopathologic-cancer-detection | 80/20 image-id split stratified by `label`, seed 42; validate ROC-AUC. |
| jigsaw-toxic-comment-classification-challenge | 80/20 split stratified by fixed composite toxicity key, seed 42; validate macro mean ROC-AUC across six labels. |
| leaf-classification | 5-fold StratifiedKFold by `species`, seed 42; validate multiclass log loss in submission class order. |
| mlsp-2013-birds | Use provided `CVfolds_2.txt`: fold 0 train, fold 1 validation, known labels only, non-empty label field means bird-present=1; validate ROC-AUC. |
| new-york-city-taxi-fare-prediction | Apply fixed invalid-row filters, then 80/20 split stratified by 10 fare quantile bins, seed 42; validate RMSE. |
| nomad2018-predict-transparent-conductors | 5-fold StratifiedKFold by combined 5-bin target quantiles, seed 42; validate mean RMSLE over two targets. |
| plant-pathology-2020-fgvc7 | 80/20 image-id split stratified by disease `idxmax`, seed 42; validate macro mean ROC-AUC across four labels. |
| random-acts-of-pizza | 5-fold StratifiedKFold by `requester_received_pizza`, seed 42, using request-time/test-available fields only; validate mean ROC-AUC. |
| ranzcr-clip-catheter-line-classification | 80/20 GroupShuffleSplit by `PatientID`, seed 13; validate macro mean ROC-AUC over submission labels. |
| siim-isic-melanoma-classification | 80/20 GroupShuffleSplit by `patient_id`, seed 42; validate ROC-AUC. |
| spooky-author-identification | 5-fold StratifiedKFold by `author`, seed 42; validate multiclass log loss in submission class order. |
| tabular-playground-series-dec-2021 | 90/10 StratifiedShuffleSplit by `Cover_Type`, seed 42; validate accuracy. |
| tabular-playground-series-may-2022 | 80/20 split stratified by `target`, seed 42; validate ROC-AUC. |
| text-normalization-challenge-english-language | Sentence-level split with `sentence_id % 20 == 0` as validation; validate exact token-level accuracy. |
| text-normalization-challenge-russian-language | Sentence-level split with `sentence_id % 20 == 0` as validation; validate exact token-level accuracy. |
| the-icml-2013-whale-challenge-right-whale-redux | 80/20 clip-level split stratified by filename `_0/_1` label, seed 42; validate ROC-AUC. |
