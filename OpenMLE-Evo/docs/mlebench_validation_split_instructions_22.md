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

原因：

真实数据有 `train.csv(id, has_cactus)` 和 `train/` 图像。这个任务的官方 metric 是 ROC-AUC，所以 validation 必须输出概率并计算 AUC，而不是 accuracy。固定 80/20 分层切分能减少正负比例波动，也比“10-20%”这种范围表述更稳定。

## 2. aptos2019-blindness-detection

Recommended prompt instruction:

```text
Validation split protocol: train.csv has columns `id_code` and ordinal class `diagnosis`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train["diagnosis"] over image ids. Train on the 80% training image ids only and validate on the 20% held-out image ids only. Apply augmentation only to training images. If the model outputs a continuous severity score, choose the four class thresholds using only the validation fold and report the resulting validation score.

Validation metric: quadratic weighted kappa between validation `diagnosis` labels and predicted integer classes 0-4. Higher is better.
```

原因：

真实数据是 `train.csv(id_code, diagnosis)`，官方 metric 是 QWK。之前没有强制 metric 时，模型可能用 accuracy、MSE 或 loss 来选节点，导致 validation 分数和 leaderboard 不一致。这里明确按 `diagnosis` 分层，并固定 QWK 作为唯一 selection metric。

## 3. denoising-dirty-documents

Recommended prompt instruction:

```text
Validation split protocol: public data contains paired dirty images in `train/` and clean targets in `train_cleaned/` with matching png filenames. For each png filename, compute `int(hashlib.md5(filename.encode()).hexdigest(), 16) % 5`; use images with value 0 as validation and all remaining images as training. If extracting patches or tiles, extract validation patches only from validation image ids and training patches only from training image ids. Never mix patches from the same original image across train and validation.

Validation metric: RMSE over pixel values on the held-out validation images reconstructed at image level. Lower is better.
```

原因：

真实数据没有 CSV 标签，而是 `train/` 与 `train_cleaned/` 成对 png。最大风险是 patch 泄漏。用 filename md5 hash 取模做 image-level 验证集，既完全确定，又比“排序后每 5 张”更不容易受文件名顺序或数据来源顺序影响。

## 4. detecting-insults-in-social-commentary

Recommended prompt instruction:

```text
Validation split protocol: train.csv has columns `Insult`, `Date`, and `Comment`. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by `Insult`. Fit text preprocessing and vectorizers separately inside each fold. Use the mean out-of-fold validation metric for model selection.

Validation metric: ROC-AUC on out-of-fold `Insult` probabilities. Higher is better.
```

原因：

真实数据是二分类文本，官方 metric 是 AUC。数据规模不大，单次 holdout seed 方差会比较明显；固定 5-fold StratifiedKFold 比随机 80/20 更稳定，同时不会让 TF-IDF/vectorizer 在验证文本上泄漏。

## 5. dog-breed-identification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has columns `id` and `breed`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train["breed"] over image ids. Train on the 80% training image ids only and validate on the 20% held-out image ids only. Apply augmentation only to training images. Use the class order from sample_submission.csv columns after `id` when building validation probability arrays.

Validation metric: multiclass log loss over all dog breed classes in sample_submission order. Lower is better.
```

原因：

真实数据有 120 个 breed 概率列，官方 metric 是 multiclass log loss。明确固定 class order 能避免因为 LabelEncoder 顺序或验证集缺类导致 log loss 不可比。

## 6. dogs-vs-cats-redux-kernels-edition

Recommended prompt instruction:

```text
Validation split protocol: parse the binary label from train image filenames: `cat.*.jpg` is 0 and `dog.*.jpg` is 1. Use train_test_split with test_size=0.2, random_state=42, and stratify by this parsed label over image filenames. Train on the 80% training images only and validate on the 20% held-out images only. Apply augmentation only to training images; validation preprocessing must be deterministic.

Validation metric: binary log loss on validation dog probabilities. Lower is better.
```

原因：

真实数据没有 train.csv，标签在 `cat.*.jpg` / `dog.*.jpg` 文件名里。官方 metric 是 log loss，不是 accuracy。固定 80/20 分层切分能避免 overconfident 模型在 accuracy 高但 log loss 差时被误选。

## 7. histopathologic-cancer-detection

Recommended prompt instruction:

```text
Validation split protocol: train_labels.csv has columns `id` and binary `label`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train_labels["label"] over image ids. Train on the 80% training image ids only and validate on the 20% held-out image ids only. Apply augmentation only to training images.

Validation metric: ROC-AUC on validation cancer `label` probabilities. Higher is better.
```

原因：

真实数据有 `train_labels.csv(id, label)` 和 `.tif` patch 图像，官方 metric 是 AUC。lite 数据里没有明确 patient/slide id，因此这里不再写“如果有 patient id”，直接固定 image-id 分层切分。

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

原因：

真实数据是 6-label multilabel 文本，官方 metric 是 macro mean ROC-AUC。之前“iterative multilabel stratification”可能因为库不可用而导致实现差异；这里改成固定 composite stratification key，所有节点都能一致执行。

## 9. leaf-classification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has target `species` and numeric features `margin1..64`, `shape1..64`, and `texture1..64`. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by `species`. Fit scalers, PCA, feature selection, and model early stopping separately inside each fold. Use the class order from sample_submission.csv columns after `id`.

Validation metric: mean multiclass log loss across the 5 validation folds, using the complete sample_submission class order. Lower is better.
```

原因：

真实数据是小规模 99 类表格/图像特征任务，官方 metric 是 multiclass log loss。固定 5-fold 比单次 holdout 更适合小数据，并且 class order 明确后，节点之间的 validation 分数才可比。

## 10. mlsp-2013-birds

Recommended prompt instruction:

```text
Validation split protocol: use the provided `essential_data/CVfolds_2.txt` and `essential_data/rec_labels_test_hidden.txt`. Treat records with `?` in `rec_labels_test_hidden.txt` as test/hidden records and exclude them from validation. For all known records, define the binary target as 1 if the label field is non-empty and 0 if the label field is empty. Train on known records with fold == 0 and validate on known records with fold == 1. If creating spectrograms, clips, frames, or segment features, create them after this recording-level split.

Validation metric: ROC-AUC on the validation binary bird-present probabilities. Higher is better.
```

原因：

真实数据自带 `CVfolds_2.txt`、`rec_id2filename.txt`、`rec_labels_test_hidden.txt`，sample_submission 只有单列 `Probability`。因此直接使用官方 fold，并把 known label 字段非空定义为 bird-present=1，空标签定义为 0，比“same target definition”更清晰。

## 11. new-york-city-taxi-fare-prediction

Recommended prompt instruction:

```text
Validation split protocol: train.csv has target `fare_amount`. First remove only these invalid training rows: `fare_amount <= 0`, `passenger_count < 1`, `passenger_count > 6`, pickup/dropoff longitude outside [-75, -72], or pickup/dropoff latitude outside [40, 42]. Do not apply any other target-dependent row filtering before the split. On the cleaned training rows, create `fare_bin = pandas.qcut(fare_amount, q=10, duplicates="drop")`. Use train_test_split with test_size=0.2, random_state=42, and stratify by `fare_bin`. Feature engineering such as haversine distance and pickup datetime features must be computed without using validation targets for training decisions.

Validation metric: RMSE on held-out `fare_amount`. Lower is better.
```

原因：

真实数据列为 `fare_amount`, pickup/dropoff 经纬度、`pickup_datetime` 和 `passenger_count`，官方 metric 是 RMSE。旧版里的 “pickup year/month or date bucket, passenger-count bucket, and approximate distance bucket if feasible” 太模糊；这里固定清洗边界和 fare decile stratification，保证长尾 fare 分布稳定，且实现简单一致。

## 12. nomad2018-predict-transparent-conductors

Recommended prompt instruction:

```text
Validation split protocol: train.csv has two regression targets: `formation_energy_ev_natom` and `bandgap_energy_ev`. Create `formation_bin = pandas.qcut(formation_energy_ev_natom, q=5, duplicates="drop")` and `bandgap_bin = pandas.qcut(bandgap_energy_ev, q=5, duplicates="drop")`. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by the combined string key `formation_bin + "_" + bandgap_bin`. Fit all feature engineering and model selection inside each fold.

Validation metric: mean RMSLE across `formation_energy_ev_natom` and `bandgap_energy_ev`, with predictions clipped to nonnegative values before RMSLE. Lower is better.
```

原因：

真实数据有两个目标，官方 metric 是两个目标 RMSLE 的平均。Nomad 的 val/test mismatch 比较明显，单次随机 holdout 容易误导；固定 5-fold target-quantile stratification 能让两个目标的分布都更稳定。

## 13. plant-pathology-2020-fgvc7

Recommended prompt instruction:

```text
Validation split protocol: train.csv has one-hot disease columns `healthy`, `multiple_diseases`, `rust`, and `scab`. Create `label_name = idxmax([healthy, multiple_diseases, rust, scab])`. Use train_test_split with test_size=0.2, random_state=42, and stratify by `label_name` over image ids. Apply augmentation only to training images.

Validation metric: macro mean ROC-AUC across the four disease probability columns. Higher is better.
```

原因：

真实数据是四列 one-hot disease label，官方 metric 是 mean column-wise ROC-AUC。固定按 label combination 分层比“80/20 或 90/10”更清晰，也能减少小类样本导致的 AUC 波动。

## 14. random-acts-of-pizza

Recommended prompt instruction:

```text
Validation split protocol: train.json has binary target `requester_received_pizza`. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by `requester_received_pizza`. Use only fields that also exist in test.json, plus request-time fields; do not use retrieval-only fields that are absent from test.json. Fit text vectorizers, encoders, and feature selection inside each fold.

Validation metric: mean ROC-AUC across the 5 folds on `requester_received_pizza` probabilities. Higher is better.
```

原因：

真实 train.json 有一些 test.json 没有的 retrieval 字段，例如 retrieval-time vote/comment fields。除了 split 不稳定外，这类字段会直接导致 test 失败或泄漏式高分。这里固定 5-fold AUC，同时明确只用 test 也有的 request-time 字段。

## 15. ranzcr-clip-catheter-line-classification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has `StudyInstanceUID`, multilabel catheter columns, and `PatientID`. Use GroupShuffleSplit with n_splits=1, test_size=0.2, random_state=13, groups=train["PatientID"]. Train on the 80% patient groups only and validate on the 20% held-out patient groups only. Use only the label columns that appear in sample_submission.csv after `StudyInstanceUID` when computing the validation metric.

Validation metric: macro mean ROC-AUC across the sample_submission label columns. Higher is better.
```

原因：

真实数据明确有 `PatientID`，所以不需要写“如果有 patient id”。医学 X-ray 任务必须按 patient 分组，避免同一病人的相关图像跨 train/val。固定 `random_state=13` 是因为在 lite 数据上它比 42 给稀有标签 `ETT - Abnormal` 留出更多验证正例，AUC 更稳定。sample_submission 只包含 9 个预测标签，metric 也应按这些提交列计算。

## 16. siim-isic-melanoma-classification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has `image_name`, `patient_id`, metadata columns, and binary target `target`. Use GroupShuffleSplit with n_splits=1, test_size=0.2, random_state=42, groups=train["patient_id"]. Train on the 80% patient groups only and validate on the 20% held-out patient groups only. Fit metadata encoders and image normalization choices only on the training fold.

Validation metric: ROC-AUC on validation melanoma `target` probabilities. Higher is better.
```

原因：

真实数据明确有 `patient_id`，官方 metric 是 AUROC。固定 patient-level split 能避免同一 patient 信息泄漏到验证集，这比普通 target stratification 更重要。

## 17. spooky-author-identification

Recommended prompt instruction:

```text
Validation split protocol: train.csv has target `author` with classes EAP, HPL, and MWS. Use 5-fold StratifiedKFold with n_splits=5, shuffle=True, random_state=42, stratified by `author`. Fit TF-IDF/tokenizers, n-gram vocabulary, SVD, and calibration inside each fold. Use class order from sample_submission.csv columns after `id`.

Validation metric: mean multiclass log loss across the 5 folds in sample_submission author order. Lower is better.
```

原因：

真实数据是三分类文本，官方 metric 是 multiclass log loss。固定 5-fold 和 submission class order 可以避免 tokenizer 泄漏和 label order 不一致。

## 18. tabular-playground-series-dec-2021

Recommended prompt instruction:

```text
Validation split protocol: train.csv has target `Cover_Type`. Use StratifiedShuffleSplit with n_splits=1, test_size=0.1, random_state=42, stratified by `Cover_Type`. Fit all preprocessing and model selection on the 90% training fold only, then evaluate once on the 10% validation fold.

Validation metric: classification accuracy on held-out `Cover_Type`. Higher is better.
```

原因：

真实数据是大规模多分类表格任务，官方 metric 是 accuracy。固定 90/10 stratified split 足够稳定且比 KFold 更省时，适合 12h agent 搜索。

## 19. tabular-playground-series-may-2022

Recommended prompt instruction:

```text
Validation split protocol: train.csv has binary target `target`. Use train_test_split with test_size=0.2, random_state=42, and stratify=train["target"]. Fit all preprocessing, `f_27` feature extraction, encoders, and model selection on the training fold only.

Validation metric: ROC-AUC on validation `target` probabilities. Higher is better.
```

原因：

真实数据列里有 `f_27`，但旧版“optionally include f_27 pattern buckets”会让不同节点采用不同 split。这里固定只按 target 分层，`f_27` 只能作为模型特征，不参与改变验证集定义。

## 20. text-normalization-challenge-english-language

Recommended prompt instruction:

```text
Validation split protocol: en_train.csv has columns `sentence_id`, `token_id`, `class`, `before`, and `after`. Convert `sentence_id` to integer. Use sentences where `sentence_id % 20 == 0` as validation, and all other sentence ids as training. Never split by token rows. Build dictionaries, rules, frequency tables, and models only from training sentences.

Validation metric: exact token-level accuracy of predicted `after` strings on validation tokens. Higher is better.
```

原因：

真实数据有 `sentence_id/token_id/class/before/after`。固定 `sentence_id % 20 == 0` 给出约 5% 句子级验证集，完全确定，避免 row-level token 泄漏。

## 21. text-normalization-challenge-russian-language

Recommended prompt instruction:

```text
Validation split protocol: ru_train.csv has columns `sentence_id`, `token_id`, `class`, `before`, and `after`. Convert `sentence_id` to integer. Use sentences where `sentence_id % 20 == 0` as validation, and all other sentence ids as training. Never split by token rows. Build dictionaries, transliteration rules, frequency tables, and models only from training sentences.

Validation metric: exact token-level accuracy of predicted `after` strings on validation tokens. Higher is better.
```

原因：

俄语文本归一化与英语一样，真实数据也有 `sentence_id/token_id/class/before/after`。固定 sentence-level modulo split 可以避免同一句话上下文泄漏，同时保证各次尝试的验证集完全一致。

## 22. the-icml-2013-whale-challenge-right-whale-redux

Recommended prompt instruction:

```text
Validation split protocol: training audio labels are encoded in train filenames ending with `_0.aif` or `_1.aif`; parse this suffix as the binary target. Use train_test_split with test_size=0.2, random_state=42, and stratify by the parsed target over original audio clip filenames. If creating spectrograms, windows, crops, or audio features, create them after this clip-level split and never mix windows from the same clip across train and validation.

Validation metric: ROC-AUC on validation right-whale probabilities. Higher is better.
```

原因：

真实 train 文件名示例是 `..._TRAIN0_0.aif`，标签在文件名后缀里。官方 metric 是 AUC。固定 clip-level stratified split 可以避免同一音频派生窗口跨 train/val。

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
