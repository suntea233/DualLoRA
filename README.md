# PromptLoRA: Zero-Shot Cross-Domain Dialogue State Tracking via Prompted Low-Rank Adaptation
Code for our Paper "**PromptLoRA**: *Zero-Shot Cross-Domain Dialogue State Tracking via Prompted Low-Rank Adaptation*"
## 🔥 Run our Code

Create a new environment with python==3.9
```shell
conda create -n promptlora python==3.9
```

Install the requirement packages
```shell
pip install -r requirements.txt
```

### 📚 Dataset
*MutliWOZ2.1*
```shell
python create_data.py
```
use create_data_2_1.py if want to run with multiwoz2.1

### 🚀 Zero-shot cross-domain

```shell
python train.py --train_batch_size 8 --gradient_accumulation_steps 8 --except_domain ${domain} --n_epochs 5
```
--except_domain: hold out domain, choose one from [hotel, train, attraction, restaurant, taxi]



