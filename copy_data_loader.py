'''
Original Copyright (c) Facebook, Inc. and its affiliates.
Licensed under Attribution-NonCommercial-ShareAlike 4.0 International License.
Modifications Copyright Ibrahim Taha Aksu, 2023.
'''

import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import os
import random
from functools import partial, reduce
from utils.fix_label import fix_general_label_error
from collections import OrderedDict, Counter, defaultdict
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
POS_SE = 0
TOTAL_SE = 0
random.seed(3407)
HISTORY_MAX_LEN = 450
GPT_MAX_LEN = 1024
os.chdir(os.path.dirname(os.path.realpath(__file__)))


class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args["slot_lang"] == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += " is " + " or ".join(item_info["value_list"]) + " or none?"
        return item_info

    def __len__(self):
        return len(self.data)


class myCollator(object):
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, data, tokenizer):
        batch_data = {}
        for key in data[0]:
            batch_data[key] = [d[key] for d in data]

        # Frozen Baseline inputs
        batch_data["fb_encoder_text"] = [
            f"{batch_data['dialog_history'][i]} {tokenizer.sep_token} {batch_data['slot_description'][i]}" for i in
            range(len(batch_data["dialog_history"]))]
        fb_batch = tokenizer(batch_data["fb_encoder_text"], padding=True, return_tensors="pt", add_special_tokens=False,
                             verbose=False)
        batch_data["fb_encoder_input"] = fb_batch["input_ids"]
        batch_data["fb_encoder_attn_mask"] = fb_batch["attention_mask"]

        # Prefix Prompter inputs
        dh_batch = tokenizer(batch_data["dialog_history"], padding=True, return_tensors="pt", add_special_tokens=False,
                             verbose=False)
        description_batch = tokenizer(batch_data["slot_description"], padding=True, return_tensors="pt",
                                      add_special_tokens=False, verbose=False)
        batch_data["encoder_input"] = dh_batch["input_ids"]
        batch_data["encoder_attention_mask"] = dh_batch["attention_mask"]
        batch_data["slot_desc_input"] = description_batch["input_ids"]
        batch_data["slot_desc_attn_mask"] = description_batch["attention_mask"]
        output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False,
                                 return_attention_mask=False)

        # replace the padding id to -100 for cross-entropy
        output_batch['input_ids'].masked_fill_(output_batch['input_ids'] == tokenizer.pad_token_id, -100)
        batch_data["decoder_output"] = output_batch['input_ids']

        return batch_data


# preprocess SGD
def read_SGD(args, path_name, tokenizer, dataset=None):
    choice_token = " <extra_id_0> "
    # read test set
    all_data = []
    # read from original data
    for filename in os.listdir(os.path.join(path_name, dataset)):
        if filename.startswith("dialogues_"):
            with open(os.path.join(path_name, dataset, filename)) as f:
                data = json.load(f)
                all_data += data
    global_tokens = []
    if dataset == "train":
        global_tokens = find_global_tokens_SGD(all_data)

    with open(os.path.join(path_name, dataset, "schema.json")) as f:
        data = json.load(f)
        descriptions = []
        check_list = ["what", "how", "whether", "which"]
        schema = {}
        for service in data:
            schema[service["service_name"]] = {}
            # collect required_slots and optional_slots
            slot_collection = []
            for intent in service["intents"]:
                for slot in intent["required_slots"]:
                    slot_collection.append(slot)
                for slot in intent["optional_slots"].keys():
                    slot_collection.append(slot)

            for slot in service["slots"]:
                description = slot["description"].lower()
                if any(c_l in description for c_l in check_list):
                    description = f"{description}?"
                else:
                    description = f"what is the {description}?"

                descriptions.append(description)

                if slot["name"] in slot_collection:
                    schema[service["service_name"]][slot["name"]] = (description, slot["possible_values"])


    schema = adjust_sgd_questions(schema)
    # desc = [value for slots in schema.values() for value in slots.values()]


    p_data = []
    # read dialogues
    for ID, dial in enumerate(all_data):
        # print(ID)
        dialog_history = ""

        for idx, turn in enumerate(dial["turns"]):
            utterance = turn["utterance"]
            utterance = fix_number(utterance)
            # User start the conversation
            if turn["speaker"] == "USER":
                assert idx % 2 == 0
                turn_belief_list = generate_belief_list(turn)

                # accumulate dialogue utterances
                # dialog_history +=  (" System: " + turn["system"] + " User: " + turn["user"])
                dialog_history += (" User: " + utterance)

                for fid, frame in enumerate(turn["frames"]):
                    # read slot values
                    for k in schema[frame["service"]]:
                        value_text = frame["state"]["slot_values"].get(k, ['none'])[0]
                        output_text = value_text + f" {tokenizer.eos_token}"
                        # for k, v in frame["state"]["slot_values"].items():
                        slot_text = k
                        question = schema[frame["service"]][k][0]
                        data_detail = {
                            "ID": dial["dialogue_id"],
                            "domains": dial["services"],
                            "domain": frame["service"],
                            "turn_id": idx,
                            "dialog_history": dialog_history,
                            "output_text": output_text,
                            "turn_belief": turn_belief_list,
                            "slot_text": slot_text,
                            "value_text": value_text,
                            "slot_domain": frame["service"],
                            "slot_description": question
                        }
                        p_data.append(data_detail)
            # system turn
            else:
                assert idx % 2 == 1
                dialog_history += (" Speaker: " + utterance)

    # with open(os.path.join("test",f"output.json"), 'w') as fout:
    #     json.dump(all_data, fout, indent=4)

    return p_data, global_tokens,descriptions


def read_MWOZ(args, path_name, SLOTS, tokenizer, description, dataset=None):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    print(("Reading all files from {}".format(path_name)))

    data = []
    domain_counter = {}
    # read files
    total = 0
    none_count = 0
    with open(path_name) as f:
        dials = json.load(f)
        global_tokens = find_global_tokens_MWOZ(dials, except_domain=args["except_domain"])

        if dataset == "train" and args["fewshot"] > 0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[:int(len(dials) * args["fewshot"])]

        for dial_dict in dials:
            dialog_history = ""

            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict[
                "domains"]) or \
                    (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict[
                        "domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                turn_id = ti

                # accumulate dialogue utterances
                dialog_history += (" System: " + turn["system"] + " User: " + turn["user"])
                if args["fix_label"]:
                    slot_values = fix_general_label_error(turn["state"]["slot_values"], SLOTS)
                else:
                    slot_values = turn["state"]["slot_values"]
                # input: dialogue history + slot
                # output: value
                slot_desc = []
                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "none":
                        slot_desc = [description[slot]['question'] for slot in slot_temp if slot in description]
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        slot_values = OrderedDict(
                            [(k, v) for k, v in slot_values.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        slot_desc = [description[slot]['question'] for slot in slot_temp if slot in description]
                        slot_values = OrderedDict(
                            [(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])

                turn_belief_list = [str(k) + '-' + str(v) for k, v in slot_values.items()]

                for slot in slot_temp:
                    # skip unrelevant slots for out of domain setting
                    if args["except_domain"] != "none" and dataset != "test":
                        if slot.split("-")[0] not in dial_dict["domains"]:
                            continue

                    value_text = slot_values.get(slot, 'none').strip()
                    output_text = value_text + f" {tokenizer.eos_token}"
                    slot_text = slot

                    none_val = int(value_text == 'none')
                    if none_val == 1:
                        none_count += 1
                    total += 1

                    if args["slot_lang"] == "human":
                        slot_lang = description[slot]["description_human"]
                    elif args["slot_lang"] == "naive":
                        slot_lang = description[slot]["naive"]
                    elif args["slot_lang"] == "value":
                        slot_lang = description[slot]["naive"]
                    elif args["slot_lang"] == "question":
                        slot_lang = description[slot]["question"]
                    elif args["slot_lang"] == "slottype":
                        slot_lang = description[slot]["slottype"]
                    else:
                        slot_lang = f"slot"

                    data_detail = {
                        "ID": dial_dict["dial_id"],
                        "domains": dial_dict["domains"],
                        "turn_id": turn_id,
                        "dialog_history": dialog_history,
                        "turn_belief": turn_belief_list,
                        "output_text": output_text,
                        "slot_text": slot_text,
                        "value_text": value_text,
                        "slot_domain": slot_text.split("-")[0],
                        "slot_description": slot_lang
                    }
                    data.append(data_detail)
    return data, slot_temp, global_tokens, slot_desc


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS


def prepare_data(args, tokenizer):
    args = vars(args)
    if args["dataset"] == "multiwoz":
        path_train = 'data/train_dials.json'
        path_dev = 'data/dev_dials.json'
        path_test = 'data/test_dials.json'

        ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
        ALL_SLOTS = get_slot_information(ontology)
        description = json.load(open("utils/slot_description.json", 'r'))

        data_train, _, global_tokens,_ = read_MWOZ(args, path_train, ALL_SLOTS, tokenizer, description, "train")
        data_dev, _, _, Dev_DESC = read_MWOZ(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
        data_test, ALL_SLOTS, _ ,Test_DESC = read_MWOZ(args, path_test, ALL_SLOTS, tokenizer, description, "test")

    elif args["dataset"] == "sgd":
        path = 'data/dstc8-schema-guided-dialogue'

        data_train, global_tokens,_ = read_SGD(args=None, path_name=path, tokenizer=tokenizer, dataset="train")
        data_dev, _, Dev_DESC = read_SGD(args=None, path_name=path, tokenizer=tokenizer, dataset="dev")
        data_test,_, Test_DESC = read_SGD(args=None, path_name=path, tokenizer=tokenizer, dataset="test")

        ALL_SLOTS = list(get_descriptions(os.path.join(path, "test", "schema.json")).keys())

    else:
        assert False, "{} is not a valid dataset name.".format(args["dataset"])

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)
    num_workers = args["worker_number"]

    collator = myCollator(args["max_size"])
    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=False,
                              collate_fn=partial(collator, tokenizer=tokenizer), num_workers=num_workers,
                              drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False,
                             collate_fn=partial(collator, tokenizer=tokenizer), num_workers=num_workers, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False,
                            collate_fn=partial(collator, tokenizer=tokenizer), num_workers=num_workers, drop_last=True)

    return train_loader, dev_loader, test_loader, ALL_SLOTS, global_tokens, Dev_DESC,Test_DESC


def adjust_sgd_questions(schema):
    if "Hotels_2" in schema:
        schema["Hotels_2"]["where_to"] = ("which city are user planning to stay in?", schema["Hotels_2"]["where_to"][1])
        schema["Hotels_2"]["has_laundry_service"] = (
        "whether the house has laundry service?", schema["Hotels_2"]["has_laundry_service"][1])
    if "Hotels_4" in schema:
        schema["Hotels_4"]["location"] = (
        "what is the city or town where the hotel is located?", schema["Hotels_4"]["location"][1])
        schema["Hotels_4"]["star_rating"] = (
        "what is the star rating of the hotel?", schema["Hotels_4"]["star_rating"][1])
        schema["Hotels_4"]["place_name"] = ("what is the name of the hotel?", schema["Hotels_4"]["place_name"][1])
    if "Media_3" in schema:
        schema["Media_3"]["genre"] = ("what type of the movie does user prefer?", schema["Media_3"]["genre"][1])
        schema["Media_3"]["starring"] = ("who is the actor in this movie?", schema["Media_3"]["starring"][1])
    if "Services_4" in schema:
        schema["Services_4"]["city"] = (
        "what is the city or area where user wants to search for a therapist?", schema["Services_4"]["city"][1])
    if "Music_3" in schema:
        schema["Music_3"]["artist"] = ("what is the name of the artist?", schema["Music_3"]["artist"][1])
        schema["Music_3"]["album"] = ("what is the album of the song?", schema["Music_3"]["album"][1])
    return schema


def generate_belief_list(turn):
    belief_list = []
    for frame in turn["frames"]:
        # read slot values
        service = frame["service"]
        for slot_name, slot_value in frame["state"]["slot_values"].items():
            belief_list.append("-".join([service, slot_name, slot_value[0]]))
    return belief_list


def get_descriptions(schema_file):
    schemas = json.load(open(schema_file))
    descriptions = {}
    for service in schemas:
        service_name = service["service_name"]
        for slot in service["slots"]:
            slot_name = slot["name"]
            slot_description = slot["description"]

            descriptions["-".join([service_name, slot_name])] = slot_description
    return descriptions


def fix_number(text):
    number_mapper = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                     "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}
    for fromx, tox in number_mapper.items():
        text = ' ' + text + ' '
        text = text.replace(f" {fromx} ", f" {tox} ")[1:-1]
    return text


def find_global_tokens_SGD(dials):
    """
        Read dialogues. Find tokens which are common except stop words.
        Sort them by frequency. i.e. most frequent non stop word should be the first element of this list.
        Use top N of these to initialize the global prompt.
    """
    print("Finding global prompts, this may take a few minutes ...")
    tokenizer = RegexpTokenizer(r'\w+')
    word_counter = Counter()
    stop_words = set(stopwords.words('english'))

    for i, dial in enumerate(dials):
        for turn in dial["turns"]:
            turn_text = turn["utterance"]
            turn_text = turn_text.translate({ord(ch): None for ch in '0123456789'}).lower()
            words = list(filter(lambda x: x not in stop_words, tokenizer.tokenize(turn_text)))
            counter = Counter(words)
            word_counter += counter

    sorted_common_words = dict(sorted(word_counter.items(), key=lambda item: item[1], reverse=True))
    print("Found global prompts")
    return sorted_common_words


def find_global_tokens_MWOZ(dials, except_domain):
    """
        Read dialogues from all 4 domains except $except_domain. Find tokens which are common across all 4 domains except stop words.
        Sort them by frequency. i.e. most frequent non stop word across all 4 domains should be the first element of this list.
        Use top N of these to initialize the global prompt.
    """

    def merge_dicts(dict1, dict2):
        dict3 = {**dict1, **dict2}
        for k, v in dict3.items():
            if k in dict1 and k in dict2:
                dict3[k] = dict1[k] + dict2[k]
        return dict3

    def find_sorted_common_words(dictionaries):
        common_keys = set(
            list(reduce(lambda x, y: set(x).intersection(set(y)), [list(dc.keys()) for dc in dictionaries.values()])))
        agg_dict = []
        for key in common_keys:
            agg_dict.append((key, min([dc[key] for dc in dictionaries.values()])))
        return sorted(agg_dict, key=lambda x: x[1], reverse=True)

    tokenizer = RegexpTokenizer(r'\w+')
    domain_word_counters = defaultdict(lambda: {})
    stop_words = set(stopwords.words('english'))

    leave_out_words = ["none", "restaurant", "train", "taxi", "hotel", "attraction", "police", "hospital"]
    stop_words = stop_words.union(set(leave_out_words))
    for dial in dials:
        words_dict = {}
        if except_domain in dial["domains"]:
            continue
        for turn in dial["turns"]:
            combined_turn = " ".join([turn["system"], turn["user"]])
            combined_turn = combined_turn.translate({ord(ch): None for ch in '0123456789'})
            words = list(filter(lambda x: x not in stop_words, tokenizer.tokenize(combined_turn)))
            counter = Counter(words)
            words_dict = merge_dicts(words_dict, counter)
        for domain in dial["domains"]:
            if domain == "police" or domain == "hospital":
                continue
            domain_word_counters[domain] = merge_dicts(domain_word_counters[domain], words_dict)
        # print('here')

    sorted_common_words = find_sorted_common_words(domain_word_counters)
    return sorted_common_words
