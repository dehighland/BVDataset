# Amsel Criteria Based Computer Vision for Diagnosing Bacterial Vaginosis
## Clue Cell Images and Amsel Criteria Values with Bacterial Vaginosis Diagnostic Labels

This page describes and links to a dataset of 30 patients tested for bacterial vaginosis (BV). For each patient, we provide slide images of vaginal microbiota that have corresponding labels of clue cell status as determined by a physician alongside Amsel criteria test values and the patient's diagnosis.

# Cite the Paper

Utilize the following citation if using this dataset:

> Highland and Zhou, Amsel criteria based computer vision for diagnosing bacterial vaginosis, Elsevier Smart Health 2024, https://doi.org/10.1016/j.smhl.2024.100501 

# Data Collection

The data was collected by a physician at Catawba Women's Center in Hickory, North Carolina between August 2022 and February 2023. The physician targeted non-pregnant female patients who self-reported vaginal discharge but did not attempt self-treatment prior to assessment. After agreement to participate in the study, a few samples of the patient's discharge were collected to perform the Amsel criteria and to send out to an outside laboratory for a NuSwab test. Data was collected from 30 patients.

The 30 patients were 20-62 year olds belonging to four racial groups: white/non-Hispanic (15 patients), white/Hispanic (4 patients), African American (9), and Asian (2 patients). According to the NuSwab test, 15 patients were BV positive (NuSwab result more than 2), 12 were BV negative (NuSwab result less than 2), and 3 had indeterminate test results (NuSwab result equal to 2). This demographic and diagnostic information is summarized in the figures below:

![DataAgeRace](https://github.com/dehighland/BV_Diagnostics/blob/9d83491366701af2609eab356710cec45bde6bf1/IMAGES/DataAgeRace.png)

In addition to collecting pH values and whiff test results for each patients, epithelial cells of each patient's discharge were imaged with a Swiftcam 18 Megapixel camera attached to a Swift SW380T 40X-2500X at a power of 40x. Across 30 patients, 3,692 cell images were captured. These 3,692 cell images were subsequently cropped into 11,181 sub-images of individual vaginal flora and assigned a binary label of clue cell status by a physician.

Our paper used the 10,024 sub-images belonging to patients with non-indeterminate NuSwab diagnoses

![DatasetSummaryTable](https://github.com/dehighland/BV_Diagnostics/blob/905b1d98ac99daf6429a84317859e67c7a4ff82d/IMAGES/DatasetSummaryTable.PNG)
