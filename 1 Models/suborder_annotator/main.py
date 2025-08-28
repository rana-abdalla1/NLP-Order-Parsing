import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config
import data_loader
import utils
from model import Model

from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa

import random
from torch.utils.data import Subset

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)
    
    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []
    
        # Initialize tqdm progress bar
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Train Epoch {epoch}")

        # Calculate the update interval (10% of the total length)
        update_interval = max(1, len(data_loader) // 10)
    
        for i, data_batch in progress_bar:
            # Update the progress bar every 10% of the total length
            if i % update_interval == 0:
                progress_bar.update(update_interval)

            data_batch = [data.cuda() for data in data_batch[:-1]]
    
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
    
            outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
    
            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
    
            loss_list.append(loss.cpu().item())
    
            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)
    
            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())
    
            self.scheduler.step()
    
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
    
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
    
        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return f1
    
    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()
    
        pred_result = []
        label_result = []
    
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
    
        # Initialize tqdm progress bar
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"{'TEST' if is_test else 'EVAL'} Epoch {epoch}")

        # Calculate the update interval (10% of the total length)
        update_interval = max(1, len(data_loader) // 10)

        with torch.no_grad():
            for i, data_batch in progress_bar:
                # Update the progress bar every 10% of the total length
                if i % update_interval == 0:
                    progress_bar.update(update_interval)

                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
    
                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length
    
                grid_mask2d = grid_mask2d.clone()
    
                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())
    
                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c
    
                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
    
                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
    
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
    
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)
    
        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))
    
        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
    
        logger.info("\n{}".format(table))
        return e_f1
    
    def predict(self, epoch, data_loader, data):
        self.model.eval()
    
        pred_result = []
        label_result = []
    
        result = []
    
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
    
        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i+config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
    
                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length
    
                grid_mask2d = grid_mask2d.clone()
    
                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())
    
                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)
    
                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c
    
                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
    
                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size
    
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
    
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)
    
        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))
    
        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
    
        logger.info("\n{}".format(table))

        # Convert the result to a DataFrame
        result_df = pd.DataFrame(result)

        # Ensure the schema matches the result structure
        schema = pa.schema([
            ('sentence', pa.string()),
            ('entity', pa.list_(pa.struct([
                ('text', pa.list_(pa.string())),  # List of strings for tokenized sentence fragments
                ('type', pa.string())  # Entity type as a string
            ])))
        ])
    
        return e_f1

    def save(self, path):
        print("Saving model to {}".format(path))
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

def predict_single_sentence(self, sentence, tokenizer, max_seq_length=512):
    """
    Predict entities for a single sentence.

    Args:
        sentence (str): The input sentence.
        tokenizer: Tokenizer initialized using the BERT model name from the configuration.
        max_seq_length (int): Maximum token sequence length for BERT (default: 512).

    Returns:
        List of predicted entities with their types.
    """
    self.model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # Step 1: Tokenize the sentence using the tokenizer
        tokens = [tokenizer.tokenize(word) for word in sentence.split()]
        pieces = [piece for sublist in tokens for piece in sublist]
        bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        bert_inputs = torch.tensor(
            [tokenizer.cls_token_id] + bert_inputs[:max_seq_length - 2] + [tokenizer.sep_token_id]
        ).unsqueeze(0).cuda()

        # Step 2: Compute `pieces2word` mapping
        length = len(sentence.split())
        pieces2word = torch.zeros((1, length, bert_inputs.size(1)), dtype=torch.bool).cuda()
        start = 0
        for i, word_pieces in enumerate(tokens):
            if word_pieces:
                piece_range = range(start + 1, start + 1 + len(word_pieces))
                pieces2word[0, i, list(piece_range)] = 1
                start += len(word_pieces)

        # Step 3: Create `dist_inputs` and `grid_mask2d`
        dist_inputs = torch.zeros((1, length, length), dtype=torch.long).cuda()
        grid_mask2d = torch.ones((1, length, length), dtype=torch.bool).cuda()

        dis2idx = np.zeros((1000), dtype='int64')
        dis2idx[1] = 1
        dis2idx[2:] = 2
        dis2idx[4:] = 3
        dis2idx[8:] = 4
        dis2idx[16:] = 5
        dis2idx[32:] = 6
        dis2idx[64:] = 7
        dis2idx[128:] = 8
        dis2idx[256:] = 9

        for i in range(length):
            for j in range(length):
                dist = j - i
                dist_index = dis2idx[abs(dist)] + (9 if dist < 0 else 0)
                dist_inputs[0, i, j] = dist_index
        dist_inputs[dist_inputs == 0] = 19

        # Step 4: Compute sentence length
        sent_length = torch.tensor([length], dtype=torch.long).cuda()

        # Step 5: Forward pass through the model
        outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

        # Step 6: Decode outputs
        predicted_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        decoded_entities = utils.decode(predicted_labels, [sentence], sent_length.cpu().numpy())

        return decoded_entities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/conll03.json')
    parser.add_argument('--save_path', type=str, default='./model.pt')
    parser.add_argument('--predict_path', type=str, default='./output.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    # Subset 10% of the training data
    train_size = int(len(datasets[0]) * 0.75)
    indices = random.sample(range(len(datasets[0])), train_size)
    train_subset = Subset(datasets[0], indices)

    # Replace datasets[0] with the subset
    datasets = (train_subset, *datasets[1:])

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                batch_size=config.batch_size,
                collate_fn=data_loader.collate_fn,
                shuffle=i == 0,
                num_workers=4,
                drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs


    logger.info("Building Model")
    model = Model(config)

    model = model.cuda()

    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        f1 = trainer.eval(i, dev_loader)
        test_f1 = trainer.eval(i, test_loader, is_test=True)
        if f1 > best_f1:
            best_f1 = f1
            best_test_f1 = test_f1
            trainer.save(config.save_path)
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load(config.save_path)
    trainer.predict("Final", test_loader, ori_data[-1])
