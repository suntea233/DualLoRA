# DualLoRA: Zero-Shot Cross-Domain Dialogue State Tracking via Dual Low-Rank Adaptations
Code for our Paper "**DualLoRA**: *Zero-Shot Cross-Domain Dialogue State Tracking via Dual Low-Rank Adaptations*"
## 🔥 Run our Code

Create a new environment with python==3.9
```shell
conda create -n duallora python==3.9
```

Install the requirement packages
```shell
pip install -r requirements.txt
```

### 📚 Dataset
*MutliWOZ*
```shell
python create_mwoz.py
```
use create_data_2_1.py if want to run with multiwoz2.1

### 🚀 Zero-shot cross-domain

```shell
python train.py --train_batch_size 8 --gradient_accumulation_steps 8 --except_domain ${domain} --n_epochs 5 --zero_initialization lora
```
--except_domain: hold out domain, choose one from [hotel, train, attraction, restaurant, taxi]



