# apprentissage-artificiel

## Installation

### Prérequis

- Python >=3.10
- pip
- virtualenv
- git


### Installation
1. Cloner le dépôt git
```bash
git clone https://github.com/mdjamina/machine_learning.git
```
2. préparer l'environement virtuel
```bash
cd machine_learning
virtualenv .venv
source .venv/bin/activate
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```
## Utilisation

### préparation des données
```bash
python ./src/pre_processing.py --datafile ./data/deft09_parlement_appr_fr.xml
```

### Entrainement
```bash
python ./src/train.py --datafile ./data/deft09_parlement_appr_fr.pkl --pipline SVC
```

### Evaluation
```bash
python ./src/evaluate.py --model ./models/model.pkl  --datatest ./data/deft09_parlement_appr_fr.xml --dataref ./data/deft09_parlement_appr_fr.xml
```


