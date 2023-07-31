# LaSCL-PLC: Lesion-Aware Supervised Contrastive Learning for Patient-Level Classification of Brainstem and Cerebellar Hemangioblastomas

## Intrduction
Brainstem and cerebellar hemangioblastoma (HB) is a rare tumor that presents risks of haemorrhage during biopsy due to the vascular nature of the lesions. Accurate preoperative diagnosis of HB is crucial for proper surgical and clinical treatment planning. However, it remains challenging for radiologists to distinguish HB from other types of intracranial tumors based solely on MRI scans, due to highly similar imaging characteristics between HB and confounding tumors.
To address this, we propose a novel patient-level classification framework leveraging lesion-aware supervised contrastive learning, named LaSCL-PLC. The core innovation is using lesion-focused supervised contrastive learning on the MRI scans to extract representations that contain rich information directly related to the tumor region itself. This prevents overfitting and provides more meaningful inputs to the classification model compared to traditional approaches. Further, by incorporating available patient demographic information such as age and gender, the model can boost classification performance.
We evaluated LaSCL-PLC on a local dataset of 240 brainstem and cerebellar tumor T1-enhanced MRI scans, including 97 scans positively identified as HB patients. Experiments demonstrate that LaSCL-PLC outperforms current state-of-the-art methods in distinguishing HB, achieving accuracy competitive with expert neuroradiologists.
By synergistically combining lesion-aware supervised contrastive learning and patient metadata, our proposed framework enables more precise preoperative classification of HB versus confounding intracranial tumors. This can dramatically enhance clinical diagnosis and surgical planning for this rare, high-risk disease. 

## main idea
![avatar](open_version/figs/fig1.png)


## how to use
### training
```python
python scripts/train_moco_cls.py -i /cinfig_path
```
### inferring
```python
python scripts/infer.py -i /cinfig_path
```

