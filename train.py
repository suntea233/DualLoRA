import argparse
import json
import os
from typing import List, Any
import logging
import torch
from pytorch_lightning import seed_everything,Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger,LightningLoggerBase,TensorBoardLogger
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, AutoConfig,T5Config
import pytorch_lightning as pl
import torch.nn as nn
import re
from evaluate import evaluate_metrics
from copy_t5 import T5ForConditionalGeneration
# from transformers import T5ForConditionalGeneration
from copy_data_loader import prepare_data
from datetime import datetime
import time

print(torch.cuda.is_available())
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",type=str,default=r"save\models\small")
    parser.add_argument("--train_batch_size",type=int,default=4)
    parser.add_argument("--worker_number", type=int, default=0, help="CPU num load for Dataloader. For win, choose 0, For Linux, choose other")
    parser.add_argument("--dev_batch_size",type=int,default=4)
    parser.add_argument("--test_batch_size",type=int,default=4)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-2)
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--dataset",type=str,default="multiwoz")
    parser.add_argument("--no_freeze", action='store_true', help="don't freeze anything")
    parser.add_argument("--model_name", type=str, default="t5", help="use t5 or bart?")
    parser.add_argument("--no_early_stop", action='store_true', help="deactivate early stopping")
    parser.add_argument("--gpu_id",type=int,default=1)
    parser.add_argument("--warm_up_steps", type=int, default=1000, help="")
    parser.add_argument("--except_domain", type=str, default="train", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--only_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--fix_label", action='store_true')
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--slot_lang", type=str, default="question", help="use 'none', 'human', 'naive', 'value', 'question', 'slottype' slot description")
    parser.add_argument("--max_size", type=int , default=250, help="Max token size of model input")
    parser.add_argument("--fewshot", type=float, default=0.0, help="data ratio for few shot experiment")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--reparametrization_ratio", type=float, default=1, help="Number of prefixes to be added")

    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5, help="Patience, use in Callback")
    # parser.add_argument('--description',type=str)

    parser.add_argument("--lora_r",type=int,default=8)
    parser.add_argument("--lora_alpha",type=int,default=16)
    parser.add_argument("--lora_dropout",type=float,default=0.05)

    parser.add_argument("--full_frozen",type=bool,default=False)
    parser.add_argument("--apply_prompt",type=bool,default=True)

    parser.add_argument("--use_prompt",action="store_true")
    parser.add_argument("--fusion_method",default="mean",type=str)
    parser.add_argument("--zero_initialization",default='linear',type=str)

    parser.add_argument("--add_reparameterization",action="store_true",help="add reparameterization trick by the original Prefix Tuning Paper. Emprically suggested only for the SGD dataset")

    parser.add_argument("--desc",default="none",type=str)


    parser.add_argument("--q",action="store_true")
    parser.add_argument("--k",action="store_true")
    parser.add_argument("--v",action="store_true")
    parser.add_argument("--o",action="store_true")


    parser.add_argument("--wandb_project_name", type=str, default="Zero_Shot_DST_T5DST_MultiWOZ_2_1_v2",
                        help="Name of the project to be displayed in wandb UI")
    parser.add_argument("--wandb_job_type", type=str, default="train", help="Job type to be displayed in wandb")
    parser.add_argument("--wandb_run_name", type=str, default="t5_run",
                        help="The name of the experiments to be displayed in wandb UI")
    parser.add_argument("--wandb_group_name", type=str, default="Standard",
                        help="Name of the experiment group to be displayed in wandb UI")
    parser.add_argument("--wandb_mode", type=str, default="offline",
                        help="Name of the experiment group to be displayed in wandb UI")

    args = parser.parse_args()
    return args


def rename_weights(key):
    match = re.match(r'^(.*\.)([qkvo]+)(\.weight)$', key)
    if match:
        prefix = match.group(1)
        letters = match.group(2)
        suffix = match.group(3)
        new_name = f'{prefix}{letters}.Con{suffix}'
        return new_name
    else:
        return key


class DST(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.ckpt = args.ckpt
        self.args = args

        self.except_domain = args.except_domain

        self.config = AutoConfig.from_pretrained(self.ckpt)
        self.config.update({"q":args.q,"k":args.k,"v":args.v,"o":args.o,"lora_r": args.lora_r,"lora_alpha":args.lora_alpha,"lora_dropout":args.lora_dropout,"apply_prompt":args.apply_prompt})

        self.weight_decay = args.weight_decay

        self.prefix_length = 10

        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt,bos_token="[bos]",eos_token="[eos]",sep_token="[sep]")


        self.model = T5ForConditionalGeneration.from_pretrained(self.ckpt,args=args)

        t5_weight = torch.load(r"save\models\small/pytorch_model.bin")
        state_dict = {rename_weights(key): value for key, value in t5_weight.items()}
        self.model.load_state_dict(state_dict=state_dict,strict=False)

        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))

        self.ff = args.full_frozen
        self.phase = 1
        self.no_freeze = args.no_freeze
        self.warm_up_steps = args.warm_up_steps

        self.lr = args.lr

        self.save_path = ""

        self.global_prompts_set = False
        self.final_global_prompt = None

        self.global_prompt = torch.nn.Parameter(data=torch.rand(self.prefix_length,
                                                                self.model.config.d_model // args.reparametrization_ratio),
                                                requires_grad=False)
        self.reparametrizer = nn.Linear(in_features=self.model.config.d_model // args.reparametrization_ratio,
                                        out_features=self.model.config.d_model)
        self.final_global_prompt = torch.nn.Parameter(data=torch.zeros(self.prefix_length,
                                                                       self.model.config.d_model),
                                                      requires_grad=False)

        self.g_q = nn.Linear(self.model.config.d_model, self.model.config.d_model, bias=False)
        self.g_k = nn.Linear(self.model.config.d_model, self.model.config.d_model, bias=False)
        self.g_v = nn.Linear(self.model.config.d_model, self.model.config.d_model, bias=False)
        self.g_o = nn.Linear(self.model.config.d_model, self.model.config.d_model, bias=False)

        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=self.model.config.hidden_size, num_heads=self.model.config.num_heads, batch_first=True)

        self.train_bs = args.train_batch_size

        self.add_reparameterization = args.add_reparameterization

    def init_global_prompt(self):
        assert not self.global_prompts_set

        initial_prompt = " ".join(pair[0] for pair in self.common_tokens)
        global_prompt_tokens = self.tokenizer(initial_prompt,
                                              padding=True,
                                              return_tensors="pt",
                                              add_special_tokens=False,
                                              verbose=False)["input_ids"][0][:self.prefix_length].cuda()
        global_prompt_data = torch.squeeze(self.model.shared(global_prompt_tokens),0)
        self.final_global_prompt.data = (global_prompt_data)

        self.global_prompts_set = True



    def training_step(self, batch, batch_idx):
        if not self.add_reparameterization and not self.global_prompts_set:
            self.init_global_prompt()

        self.model.train()
        encoder_input = batch['fb_encoder_input'].cuda()
        encoder_attn_mask = batch["fb_encoder_attn_mask"].cuda()

        prompt_embed = self.model.shared(batch["slot_desc_input"].cuda())
        prompt_embed_mask = batch['slot_desc_attn_mask']

        expanded_prompt = torch.unsqueeze(self.final_global_prompt, 1).expand(-1, self.train_bs, -1)

        expanded_prompt = expanded_prompt.transpose(0,1)

        query = self.g_q(expanded_prompt.cuda())
        key = self.g_k(prompt_embed)
        value = self.g_v(prompt_embed)

        attention_out = self.cross_attn(query=query,key=key,value=value,key_padding_mask=~prompt_embed_mask.to(bool))

        attended_prompt = self.g_o(attention_out[0])


        model_output = self.model(
            input_ids=encoder_input,
            attention_mask=encoder_attn_mask,
            labels=batch['decoder_output'],
            prompt_embed=prompt_embed,
            prompt_embed_mask=prompt_embed_mask,
            hidden_attention_mask=encoder_attn_mask,
            global_prompt=attended_prompt
        )

        loss = model_output['loss']
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss


    def on_validation_epoch_start(self) -> None:
        tokenized_desc = self.tokenizer(self.dev_desc,padding=True,return_tensors='pt')
        input_ids = tokenized_desc['input_ids']

        attention_mask = tokenized_desc['attention_mask']

        prompt_embed = self.model.shared(input_ids.cuda())
        prompt_embed_mask = attention_mask.cuda()
        eos_token_id = self.tokenizer.eos_token_id

        batch_size = prompt_embed.size(0)

        expanded_prompt = torch.unsqueeze(self.final_global_prompt, 1).expand(-1, batch_size, -1)
        expanded_prompt = expanded_prompt.transpose(0,1)

        query = self.g_q(expanded_prompt.cuda())
        key = self.g_k(prompt_embed)
        value = self.g_v(prompt_embed)

        attention_out = self.cross_attn(query=query,key=key,value=value,key_padding_mask=~prompt_embed_mask.to(bool))

        global_prompt = self.g_o(attention_out[0])

        named_state_dict = self.model.named_parameters()

        lora_con_A = []
        lora_con_B = []


        for name,param in named_state_dict:
            if "lora_con_A" in name:
                lora_con_A.append(param)
            if "lora_con_B" in name:
                lora_con_B.append(param)

        # global_prompt @ self.lora_con_A.transpose(0, 1) @ self.lora_con_B.transpose(0, 1)
        p_bias = []

        for i in range(len(lora_con_A)):
            p_result = (global_prompt @ lora_con_A[i].transpose(0, 1) @ lora_con_B[i].transpose(0,1))
            p_bias.append(p_result.tolist())

        self.p_bias = torch.tensor(p_bias).transpose(0,1)


    def validation_step(self, batch, batch_idx):
        self.model.eval()
        encoder_input = batch['fb_encoder_input'].cuda()
        encoder_attn_mask = batch["fb_encoder_attn_mask"].cuda()

        index = [self.dev_desc.index(desc) for desc in batch['slot_description']]

        p_bias = self.p_bias[index].cuda()

        model_output = self.model(
            input_ids=encoder_input,
            attention_mask=encoder_attn_mask,
            labels=batch['decoder_output'],
            p_bias=p_bias,
        )

        loss = model_output['loss']

        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss



    def test_epoch_end(
        self, outputs: List[Any]
    ) -> None:

        prefix = "zero-shot"
        save_path = os.path.join(self.save_path, "results")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for slot_log in self.slot_logger.values():
            slot_log[2] = (slot_log[1] / slot_log[0]) if slot_log[0] != 0 else 0

        joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(self.predictions, self.all_slots)

        evaluation_metrics = {"Joint Acc": joint_acc_score, "Turn Acc": turn_acc_score, "Joint F1": F1_score}
        slot_acc_metrics = {key: value[2] for (key, value) in self.slot_logger.items()}
        self.wandb_logger.log_metrics(metrics=slot_acc_metrics)
        self.wandb_logger.log_metrics(metrics=evaluation_metrics)

        now = datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M")


        with open(os.path.join(save_path, f"{prefix}_result_{now}.json"), 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)


        print(f"{prefix} result:", evaluation_metrics)

        with open(os.path.join(save_path, f"{prefix}_prediction_{now}.json"), 'w') as f:
            json.dump(self.predictions, f, indent=4)

        self.slot_logger = {slot_name: [0, 0, 0] for slot_name in self.all_slots}
        self.predictions = {}

        self.log("Joint Acc", joint_acc_score, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("Joint Acc", joint_acc_score, on_step=True, on_epoch=False, prog_bar=True)


    def test_step(self, batch, batch_idx):
        self.model.eval()
        dst_outputs = self.generate(batch)
        value_batch = self.tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        for idx, value in enumerate(value_batch):
            # For some reason the new generation adds a trailing whitespace to each decoded value
            # This is my naive solution.
            value = value.strip()

            dial_id = batch["ID"][idx]
            if dial_id not in self.predictions:
                self.predictions[dial_id] = {}
                self.predictions[dial_id]["domain"] = batch["domains"][idx][0]
                self.predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in self.predictions[dial_id]["turns"]:
                self.predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief": batch["turn_belief"][idx],
                                                                        "pred_belief": []}

            if self.args.dataset == "sgd":
                pred_slot = str(batch["domain"][idx]) + '-' + str(batch["slot_text"][idx])
            else:
                pred_slot = str(batch["slot_text"][idx])

            if value != "none":
                self.predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(pred_slot + '-' + str(value))

            # analyze slot acc:
            if str(value) == str(batch["value_text"][idx]):
                self.slot_logger[pred_slot][1] += 1  # hit
            self.slot_logger[pred_slot][0] += 1  # total



    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        using_native_amp,
        using_lbfgs,
    ):
        if self.ff:
            optimizer.step(closure=optimizer_closure)
        else:
            if self.phase == 1:
                if self.global_step >= self.warm_up_steps and not self.no_freeze:
                    self.phase = 2
                    print("Second phase start")

                    for param in self.model.parameters():
                        param.requires_grad = False

                    for param in self.model.encoder.block[0].parameters():
                        param.requires_grad = True

                    for param in self.model.encoder.block[-1].parameters():
                        param.requires_grad = True

                    for param in self.model.lm_head.parameters():
                        param.requires_grad = True

                    for name,param in self.model.named_parameters():
                        if "lora_" in name:
                            param.requires_grad = True

                # If optimizer_idx == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                # update params
                optimizer.step(closure=optimizer_closure)


    def generate(self,batch):
        self.model.eval()
        encoder_input = batch['fb_encoder_input'].cuda()
        encoder_attn_mask = batch["fb_encoder_attn_mask"].cuda()

        prompt_embed = self.model.shared(batch["slot_desc_input"].cuda())
        prompt_embed_mask = batch['slot_desc_attn_mask'].cuda()
        eos_token_id = self.tokenizer.eos_token_id

        batch_size = encoder_input.shape[0]
        expanded_prompt = torch.unsqueeze(self.final_global_prompt, 1).expand(-1, batch_size, -1)
        expanded_prompt = expanded_prompt.transpose(0,1)

        query = self.g_q(expanded_prompt.cuda())
        key = self.g_k(prompt_embed)
        value = self.g_v(prompt_embed)

        attention_out = self.cross_attn(query=query,key=key,value=value,key_padding_mask=~prompt_embed_mask.to(bool))

        attended_prompt = self.g_o(attention_out[0])


        dst_output = self.model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_attn_mask,
            eos_token_id=eos_token_id,
            prompt_embed=prompt_embed,
            prompt_embed_mask=prompt_embed_mask,
            hidden_attention_mask=encoder_attn_mask,
            global_prompt=attended_prompt,
            max_length=40,
            # num_beams=5,
            # early_stopping=True,
        )

        return dst_output


    def configure_optimizers(self):
        return AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,correct_bias=True,weight_decay=self.weight_decay)
        # return AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,correct_bias=True)


def train(args):

    from datetime import datetime

    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M")

    model = DST(args)

    # save_path = os.path.join(args.saving_dir, args.model_name + "-" + now + "-" + args.except_domain)
    if args.desc != "none":
        save_path = os.path.join(args.saving_dir, args.model_name + "-" + now + "-" + args.except_domain + '-' + args.desc)
    else:
        save_path = os.path.join(args.saving_dir, args.model_name + "-" + now + "-" + args.except_domain)


    model.save_path = save_path

    run_name = args.wandb_run_name
    wandb_logger = WandbLogger(name= run_name,project=args.wandb_project_name,job_type=args.wandb_job_type ,group=args.wandb_group_name)
    model.wandb_logger = wandb_logger

    seed_everything(args.seed)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_args(args,save_path)

    train_loader, val_loader, test_loader, all_slots, global_prompts,dev_desc,test_desc = prepare_data(args, model.tokenizer)

    model.common_tokens = global_prompts
    model.all_slots = all_slots
    # model.all_desc = all_desc
    model.dev_desc = dev_desc
    model.test_desc = test_desc

    model.slot_logger = {slot_name: [0, 0, 0] for slot_name in all_slots}
    model.predictions = {}

    checkpoint_callback = ModelCheckpoint(filepath= save_path+"/{epoch}-{global_step}-{Joint Acc:.3f}",
                                          monitor='Joint Acc',
                                          verbose = False,
                                          save_last= True,
                                          save_top_k = 1,
                                          mode="max",
                                          )


    callbacks = []
    if not args.no_early_stop:
        callbacks= [pl.callbacks.EarlyStopping(monitor='Joint Acc',
                                               min_delta=args.min_delta,
                                               patience=args.patience,
                                               verbose=False,
                                               mode='max')]

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    gradient_clip_val=args.max_norm,
                    max_epochs=args.n_epochs,
                    callbacks=callbacks,
                    checkpoint_callback = checkpoint_callback,
                    deterministic=True,
                    num_nodes=1,
                    #precision=16,
                    logger = wandb_logger,
                    gpus=1,
                    val_check_interval = 1.0,
                    # num_sanity_val_steps=1,

                    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model,test_loader)



def save_args(args,save_path):
    argsDict = args.__dict__
    with open(save_path + '\\args.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')



def main():
    args = parse_args()

    os.environ['WANDB_MODE'] = args.wandb_mode
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    train(args)


if __name__ == "__main__":
    print(torch.cuda.device_count())
    main()
