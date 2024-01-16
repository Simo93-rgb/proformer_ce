# Proformer

Proformer is a transformer based model for process traces presented in"[Structural Positional Encoding for Knowledge Integration in Transformer-based Medical Process Monitoring](https://ceur-ws.org/Vol-3578/paper3.pdf)". Here we present the source code applied to the [BPI 2012 challenge](10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb) dataset.

## Performance

|  BPI20212   |     SPE     |     No SPE    |
|-------------|-------------|---------------|
| **1**       | 0.8140      | 0.8601        |
| **3**       | 0.9742      | 0.9800        |
| **5**       | 0.9925      | 0.9939        |



## Installation

To install the required library using pip:

```bash
git clone https://gitlab.di.unipmn.it/Christopher.Irwin/proformer.git
cd proformer
pip install -r requirements.txt
```

## Usage

To run a training using the best parameters use:

```python
# run on BPI2012 dataset using SPE
python run_proformer.py --use_taxonomy
# run on BPI2012 dataset WITHOUT SPE    
python run_proformer.py 
```

## Examples

The `notebooks` directory contains the dataset preprocessing. In general, it is sufficient to have a csv containing a `case_id` column representing a unique identifier for the cases and a `activity` column representing the applied actions.

## Contacts

- christopher.irwin@uniupo.it

