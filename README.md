# LEADER: A Hierarchy-Aware Drug Recommendation Framework by Fusing Molecular Knowledge and Electronic Health Record

Electronic Health Records (EHR)-based medication recommendation models have demonstrated significant clinical value in promoting precision medicine and enabling personalized treatment. However, existing approaches face two major challenges: **insufficient integration of drug molecular knowledge** and **inadequate handling of hierarchical medication taxonomies**. 

To address these challenges, this study proposes the **LEADER (Large Language Model Enhanced Medication Recommendation)** framework, which achieves accurate and interpretable medication prediction through hierarchical prediction and cross-attention drug knowledge fusion. Specifically, this method first performs LoRA fine-tuning of LLaMA-7B on clinical narratives with hierarchical ATC labels. Subsequently, a cross-attention mechanism integrates molecular drug embeddings to enhance drug-specific representations. Finally, a 4-layer defense strategy addresses severe class imbalance in medication data.

This innovative solution not only greatly improves prediction accuracy but also provides interpretable hierarchical predictions and validated causal reasoning capability, offering a practical technical approach for clinical decision support systems.

## Framework

<img width="1980" height="1057" alt="fig1" src="https://github.com/user-attachments/assets/f8a7acaa-bfe1-47a2-a0f5-88310fd2d784" />


## Data Format

All data files are in **JSONL** format (one JSON object per line). All datasets used in this project are derived from the **MIMIC-III** clinical database.

We provide the complete data preprocessing pipeline. The processed datasets follow a hierarchical structure aligned with the ATC classification system.


### `train_atc_hierarchy2.json` / `val_atc_hierarchy2.json` / `test_atc_hierarchy2.json`

Each line is a training sample:

```json
{
  "input": "Currently, in the most recent visit, the patient presents with Diabetes mellitus type II, Hypertension, Hyperlipidemia and receives the following procedures: None. Based on these findings, the recommended medications are: ",
  "target": "insulins and analogues, beta blocking agents, ace inhibitors, plain, lipid modifying agents, plain",
  "subject_id": 12345,
  "atc_level_1": ["A", "C"],
  "atc_level_2": ["A1", "C0", "C1"],
  "atc_level_3": ["A10", "C07", "C09", "C10"],
  "atc_level_4": ["A10A", "C07A", "C09A", "C10A"]
}
```

### `drug_embeddings.pt`

PyTorch file storing molecular drug embeddings, keyed by ATC code:

```python
# {"atc4_embeddings": Tensor(num_drugs, 384), "atc4_to_idx": {atc: idx}, ...}
# Generated from drug SMILES strings with ChemBERTa (DeepChem/ChemBERTa-77M-MLM)
drug_data = torch.load('data/mimic3/l_data/drug_embeddings.pt')
```

---

### Step 1: Download External Resources

`download.sh` fetches everything that is not stored in this repository and puts it where the code expects it. Existing files are skipped.

```bash
bash download.sh aux         # auxiliary mapping files + data/mimic3/utils.py
bash download.sh models      # LLaMA-7B and ChemBERTa -> resources/
bash download.sh embeddings  # data/mimic3/l_data/drug_embeddings.pt
```

- `drug-DDI.csv` and `drugbank_drugs_info.csv` are hosted on Google Drive (links from [SafeDrug](https://github.com/ycq091044/SafeDrug)); install `gdown` or download them manually into `data/mimic3/auxiliary/`.
- LLaMA-7B and MIMIC-III are licensed resources and must never be committed to this repository.

### Step 2: Download MIMIC-III (only needed to rebuild the dataset)

Apply for access at [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/), then either run

```bash
PHYSIONET_USER=<your_username> bash download.sh mimic
```

or unzip `PRESCRIPTIONS`, `DIAGNOSES_ICD`, `PROCEDURES_ICD`, `ADMISSIONS`, `D_ICD_DIAGNOSES`, `D_ICD_PROCEDURES` (`.csv`) into `data/mimic3/raw/`.

### Step 3: Preprocess Data (optional)

```bash
python data_process.py
```

Run from the repository root. Intermediate files go to `data/mimic3/handled/`, datasets and drug embeddings to `data/mimic3/l_data/`.

### Step 4: Train Model with LoRA

```bash
bash experiment/llm_cls.bash
```

The script trains, then predicts on the test and validation sets (`results/atc_hierarchy2/`, `results/atc_hierarchy2_val/`). Paths can be overridden with `MODEL_NAME_OR_PATH=... DATA_DIR=...`.

---

### Step 5: Evaluate on Test Set

```bash
python final_evaluation_with_bootstrap.py
```

All input paths have defaults matching the layout below and can be changed with command-line flags (`--help`).



## Data Preparation

### File Structure

```text
data/mimic3/
├── raw/                           # MIMIC-III tables (not in git, PhysioNet license)
│   ├── PRESCRIPTIONS.csv  DIAGNOSES_ICD.csv  PROCEDURES_ICD.csv
│   └── ADMISSIONS.csv  D_ICD_DIAGNOSES.csv  D_ICD_PROCEDURES.csv
├── auxiliary/                     # download.sh aux (not in git)
│   ├── drug-DDI.csv               # drug-drug interactions (Google Drive, via SafeDrug)
│   ├── drug-atc.csv               # CID -> ATC
│   ├── ndc2RXCUI.txt              # NDC -> RXCUI
│   ├── RXCUI2atc4.csv             # RXCUI -> ATC
│   ├── drugbank_drugs_info.csv    # drug name -> SMILES (Google Drive, via SafeDrug)
│   └── WHO ATC-DDD 2021-12-03.csv # ATC code -> drug name
├── utils.py                       # preprocessing helpers (download.sh aux, from LEADER)
├── construction.py                # legacy LEADER-format preprocessing
├── handled/                       # intermediate outputs (not in git)
│   ├── atc3toSMILES.pkl
│   ├── ddi_A_final.pkl
│   └── atc4_vocab.pkl             # optional, used by the bootstrap DDI evaluation
└── l_data/
    ├── train_atc_hierarchy2.json  # training set
    ├── val_atc_hierarchy2.json    # validation set
    ├── test_atc_hierarchy2.json   # test set
    └── drug_embeddings.pt         # molecular drug embeddings
resources/                         # model weights (not in git)
├── llama-7b/
└── ChemBERTa-77M-MLM/
saved/lora-atc_hierarchy2/checkpoint-*/   # LoRA checkpoints (not in git)
results/                                  # predictions (not in git)
```



### Project Structure

```text
├── data/
│   └── mimic3/
│       ├── raw/
│       ├── auxiliary/
│       ├── handled/
│       └── l_data/
├── llm/
│   ├── llama.py                       # LLaMA with hierarchical head
│   ├── drug_knowledge_module.py       # Cross-attention module
│   ├── peft/                          # LoRA implementation
│   └── data_processor/
├── generators/
│   └── data.py                        # EHRTokenizer, vocabulary
├── experiment/
│   └── llm_cls.bash                   # Training script
├── saved/
│   └── lora-atc_hierarchy2/
│       └── checkpoint-5000/           # Best checkpoint
├── results/
├── download.sh                        # Fetch external resources
├── data_process.py                    # Preprocessing pipeline
├── main_llm_cls.py                    # Main training/inference
├── final_evaluation_with_bootstrap.py # Evaluation
└── README.md
```





















