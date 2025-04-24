### Install

From the root folder, to be in dev mode:
```bash
pip install -e .
```

Run data augmentation helper and follow the instructions (this will take a bit of time ~15min):
```bash
python biometrics/last_q/scripts/data_augmentation.py -h
usage: data_augmentation.py [-h] casia1_dir nist301_dir

Augment and save NIST301 fingerprint images and CASIA1 iris images

positional arguments:
  casia1_dir   Path to the CASIA1 dataset folder
  nist301_dir  Path to the NIST301 dataset folder

options:
  -h, --help   show this help message and exit
```

Then run the training of model by running the command:
```bash
python biometrics/last_q/scripts/$script_name.py
```
where `$script_name` is the script name like `train_from_scratch.py`

/!\ MAKE SURE TO ADAPTH THE PATH OF THE INPUT DATA

