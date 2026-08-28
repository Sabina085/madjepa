# MADJEPA - Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection

Official implementation of the MADJEPA framework.

## Method


<img src="./figures/ijepa-pipeline.png" alt="drawing" width="500"/>


<!-- ![image](./figures/ijepa-pipeline.png) -->


IJEPA paper link
https://arxiv.org/pdf/2301.08243


BMAD benchmark link
https://arxiv.org/abs/2306.11876


## Setup

## MADJEPA training pipeline

To train the full MADJEPA pipeline, run the following command:

```bash
python3 training_pipeline.py \
 --train_ijepa_config_pattern_path="configs/train_ijepa/<DATASET_NAME>.yaml"
```

### Examples
* <b> Brats2021 (Brain MRI)</b> dataset : <code> python3 training_pipeline.py --train_ijepa_config_pattern_path="configs/train_ijepa/BraTS2021_slice.yaml"</code>

* <b> hist_DIY (Liver CT) </b> dataset : <code> python3 training_pipeline.py --train_ijepa_config_pattern_path="configs/train_ijepa/hist_DIY.yaml"</code>

* <b> RSNA (Chest X-ray) </b> dataset : <code> python3 training_pipeline.py --train_ijepa_config_pattern_path="configs/train_ijepa/ChestRSNA.yaml"</code>

This command will <b>sequentially run the three training stages</b>:
1. <b>I-JEPA training</b> (stage 1)
2. <b>Target encoder distillation</b> (stage 2)
3. <b>Pseudo-anomaly classifier</b> (stage 3)


### CLI arguments
* <code> train_ijepa_config_pattern_path</code>  = path to the I-JEPA training yaml config file. The configuration files that we used for each dataset could be found in the <code>configs/train_ijepa/</code> folder.


#### &nbsp;&nbsp;&nbsp;&nbsp;Notes

 &nbsp;&nbsp;&nbsp;&nbsp;The configs for target encoder distillation and the pseudo-anomaly classifier could be found in the folders <code> configs/train_distillation </code> and <code>configs//tsynthetic_anomaly_classifier</code> respectively.

 &nbsp;&nbsp;&nbsp;&nbsp; The command will automatically read the configuration files having the same name as <code>train_ijepa_config_pattern_path</code> for phase 2 and phase 3 from the aforementioned folders.


# Anomaly detection pipeline
For running complete anomaly detection pipeline with MADJEPA, use the following command:

```bash
python3 anomaly_detection_pipeline.py \
 --inference_config_pattern_path="configs/inference_anomaly_detection/<DATASET_NAME>.yaml" \
 --weight_distillation_test=<WEIGHT_DISTILLATION> \
 --weight_classification_test=<WEIGHT_CLASSIFICATION> \
```

### Examples
<b> Brats2021 (Brain MRI)</b> dataset :
```bash
python3 anomaly_detection_pipeline.py \
--inference_config_pattern_path="configs/inference_anomaly_detection/BraTS2021_slice.yaml" \
--weight_distillation_test=0.5 \
--weight_classification_test=0.1 
```

## Acknowledgements
This work is built on the code and concepts from the [I-JEPA codebase](https://github.com/facebookresearch/ijepa). 

