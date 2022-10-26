import torch
import random
import itertools
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from transformers import BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
import time
#from torch.utils.tensorboard import SummaryWriter
from build_graph import generate_text_graph
from GCN import gcn

from sklearn.metrics import f1_score, accuracy_score

def kl_div(param1, param2):
    mean1, cov1 = param1
    mean2, cov2 = param2
    bsz, seqlen, tag_dim = mean1.shape
    var_len = tag_dim * seqlen

    cov2_inv = 1 / cov2
    mean_diff = mean1 - mean2

    mean_diff = mean_diff.view(bsz, -1)
    cov1 = cov1.view(bsz, -1)
    cov2 = cov2.view(bsz, -1)
    cov2_inv = cov2_inv.view(bsz, -1)

    temp = (mean_diff * cov2_inv).view(bsz, 1, -1)
    KL = 0.5 * (torch.sum(torch.log(cov2), dim=1) - torch.sum(torch.log(cov1), dim=1) - var_len
                + torch.sum(cov2_inv * cov1, dim=1) + torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz))
    return KL

def random_word(tokens, tokenizer, select_prob=0.3):

    output_label = []
    tokens = list(tokens)

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < select_prob:
            prob /= select_prob

            # 80% randomly change token to mask token
            if prob < 1.0:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.get_vocab().items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.get_vocab()[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.get_vocab()["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-100)

    return tokens, output_label

def preprocess(support_set,
               query_set,
               labels,
               additional,
               tokenizer):
    input_args = {}

    input_args['support_inputs'] = [
        torch.tensor([lst[:128] for lst in tokenizer(sp, padding='max_length', max_length=128)['input_ids']]).type(
            torch.LongTensor) for sp in support_set]

    query_labels = additional['query_labels']

    input_args['support_mask'] = [(e != 0).type(torch.cuda.LongTensor) for e in input_args['support_inputs']]

    input_args['query_inputs'] = torch.tensor(
        [lst[:128] for lst in tokenizer(query_set,padding='max_length', max_length=128)['input_ids']]).type(
        torch.cuda.LongTensor)
    input_args['query_mask'] = (input_args['query_inputs'] != 0).type(torch.cuda.LongTensor)

    input_args['labels'] = torch.tensor(labels).type(torch.cuda.LongTensor)
    input_args['query_labels'] = torch.tensor(query_labels).type(torch.cuda.LongTensor)

    if len(additional) != 0 and len(additional) == 4:
        input_args['head_sup'] = torch.cuda.LongTensor(additional['head_sup'])
        input_args['tail_sup'] = torch.cuda.LongTensor(additional['tail_sup'])
        input_args['head_que'] = torch.cuda.LongTensor(additional['head_que'])
        input_args['tail_que'] = torch.cuda.LongTensor(additional['tail_que'])

    return input_args


class CompressedDist(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim):
        super(CompressedDist, self).__init__()
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(input_dim, (input_dim + output_dim) // 2),
            torch.nn.ReLU(),
            torch.nn.Linear((input_dim + output_dim) // 2, output_dim),
            torch.nn.ReLU())
        '''
        for param in self.hidden.parameters():
            torch.nn.init.normal_(param, mean=.0, std=.1)
        '''
        self.mean_vec = torch.nn.Linear(output_dim, output_dim)
        self.std_vec = torch.nn.Linear(output_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, reps, n_samples=1):
        hidden = self.hidden(reps)
        mean = self.mean_vec(hidden)
        std = self.std_vec(hidden) ** 2 + 1e-8
        noise = torch.randn(n_samples, *reps.shape[:-1], self.output_dim).type(torch.cuda.FloatTensor)
        samples = std * noise + mean

        return samples.transpose(0, 1), (mean, std)


class CTNet(torch.nn.Module):
    def __init__(self,
                 bert_type='bert-base-uncased',
                 metric='distance',
                 gamma=1e-2,
                 N=5,
                 K=1, l=0.0, p = 0.0):
        super(CTNet, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_type)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        null_emb = torch.FloatTensor(np.random.uniform(-0.01, 0.01, (6, 768)))
        null_emb = torch.nn.parameter.Parameter(null_emb)

        emb_cat = torch.nn.parameter.Parameter(
            torch.cat([self.bert.embeddings.word_embeddings.weight, null_emb], dim=0))

        self.bert.embeddings.word_embeddings.weight = emb_cat

        token_dict = {'additional_special_tokens': ['[H]', '[/H]', '[T]', '[/T]', '[F]', '[/F]']}
        self.tokenizer.add_special_tokens(token_dict)

        self.bert = torch.nn.DataParallel(self.bert)
        self.ls = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.bce_loss = torch.nn.BCELoss(reduction='none')
        self.nll = torch.nn.NLLLoss()
        self.relu = torch.nn.ReLU()
        self.count = 0
        self.maml = False
        self.rel_net = False
        self.decompose = False
        self.multi_label = False
        self.case_based = False

        self.cnn = None
        self.train_acc_list = []
        self.extract_diff = diff_extractor
        self.metric = metric
        self.sw = False
        self.gamma = gamma
        self.inner_opt = None
        self.inner_sch = None

        self.linear1 = torch.nn.Linear(768, N)

        self.l = l
        self.p = p

    def atten_fusion(self, inputs):

        t = inputs.shape[0]

        hidden_size = inputs.shape[-1]

        # 5 x 5 x 768
        # inputs = inputs.permute(1, 0, 2)

        outputs = []
        for x in range(t):
            # [5 x 768]
            t_x = inputs[x, :, :]
            # [5 768] = [5 768] x [768, 768]
            x_proj = torch.matmul(t_x, self.atten_fc[x])
            x_proj = torch.nn.Tanh()(x_proj)
            u_w = self.atten_fc2[x]
            # [5 1] = [5 768] x [768 1]
            x = torch.matmul(x_proj, u_w)
            alphas = torch.nn.Softmax(dim=-1)(x)

            # [768 1] = [768 5] x [5 1]
            output = torch.matmul(t_x.permute(1, 0), alphas)
            output = torch.squeeze(output, dim=-1)
            outputs.append(output)

        final_output = torch.stack(outputs, dim=0)
        # final_output = self.classess(final_output)
        return final_output

    def get_ib_params(self):
        ret = []
        if not self.decompose:
            ret += list(self.fnn.parameters())
        elif self.decompose:
            ret += list(self.decomposer.parameters())
        return ret

    def get_bert_params(self):
        return list(self.bert.parameters())

    def switch(self):
        self.sw = (not self.sw)

    def forward(self, support_set, query_set, label, additional):
  
        kwargs = preprocess(support_set, query_set, label, additional, self.tokenizer)
        N = len(kwargs['support_inputs'])
        K = kwargs['support_inputs'][0].shape[0]
        Q = len(kwargs['query_inputs'])
        labels = kwargs['labels']

        if len(labels.shape) > 1:
            labels = labels.reshape((-1,))

        support_set_cat = torch.cat(kwargs['support_inputs'], dim=0).cuda()

        support_mask = (support_set_cat != 0).type(torch.LongTensor).cuda()
        query_set = kwargs['query_inputs']
        query_mask = (query_set != 0).type(torch.LongTensor).cuda()

        bert_out = self.bert(input_ids=support_set_cat, attention_mask=support_mask)

        local_support_rep = bert_out[0]
        support_rep = local_support_rep[:, 0]
        lm_logits = self.linear1(support_rep).cuda()
        lm_pred = self.ls(lm_logits)
        lm_label = torch.tensor([_ for u in range(K) for _ in range(N)]).cuda()
        lm_loss = self.nll(lm_pred, lm_label).mean()
        Th = 768

        support_rep = support_rep.reshape((-1, K, support_rep.shape[-1]))

        if K <= 1:
            support_et = support_rep
        else:
            N = support_rep.shape[1]
            dar = list(itertools.combinations(range(support_rep.shape[1]), 2))
            support_et = torch.zeros(support_rep.shape[0], K, Th).float().cuda()

            for i in range(support_rep.shape[0]):
                for j in range(K):
                    support_et[i, j, :] += (support_rep[i, dar[j][0] , :] + support_rep[i, dar[j][1], :]) / 2

        support_rep = support_rep.mean(dim=1)

        local_query_rep = self.bert(input_ids=query_set, attention_mask=query_mask)[0]
        query_rep = local_query_rep[:, 0]

        if K <= 1:
            tar = list(itertools.combinations(range(support_rep.shape[0]), 2))
            dd = 0.0
            for i in tar:
                tmp = ((support_rep[i[0]] - support_rep[i[1]]) ** 2).sum(dim=-1) ** 0.5
                dd += tmp
            dd_loss = len(tar) / dd

        logits2 = torch.zeros(query_rep.shape[0], support_et.shape[0]).float().cuda()
        if K > 1:
            for i in range(query_rep.shape[0]):
                for j in range(support_et.shape[0]):
                    for x in range(support_et.shape[1]):
                        logits2[i][j] += -((support_et[j, x, :] - query_rep[i]) ** 2).sum(dim=-1) ** 0.5
                    logits2[i][j] /= support_et.shape[1]

        support_rep = support_rep.unsqueeze(0)
        query_rep = query_rep.unsqueeze(1)
        logits = -((support_rep - query_rep) ** 2).sum(dim=-1) ** 0.5
        pred = self.ls(logits)
        if K > 1:
            pred2 = self.ls(logits2)
            res = pred * self.p + (1.0 - self.p) * pred2
            acc =  accuracy_score(labels.cpu().numpy(), res.argmax(dim=-1).cpu().numpy().tolist())
            loss = self.nll(res, labels).mean()
        else:
            acc =  accuracy_score(labels.cpu().numpy(), pred.argmax(dim=-1).cpu().numpy().tolist())
            loss = self.nll(pred, labels).mean()  * dd_loss
        self.train_acc_list.append(acc)

        self.count += 1

        return loss + self.l * lm_loss 

    def predict(self, support_set, query_set, additional):
        kwargs = preprocess(support_set, query_set, [], additional, self.tokenizer)
        N = len(kwargs['support_inputs'])
        K = kwargs['support_inputs'][0].shape[0]
        Q = len(kwargs['query_inputs'])

        support_set_cat = torch.cat(kwargs['support_inputs'], dim=0)
        support_mask = (support_set_cat != 0).type(torch.LongTensor).cuda()
        query_set = kwargs['query_inputs']
        query_mask = (query_set != 0).type(torch.LongTensor).cuda()

        bert_out = self.bert(input_ids=support_set_cat, attention_mask=support_mask)

        local_support_rep = bert_out[0]
        support_rep = local_support_rep[:, 0]

        Th = 768

        support_rep = support_rep.reshape((-1, K, support_rep.shape[-1]))

        if K <= 1:
            support_et = support_rep
        else:
            N = support_rep.shape[1]
            dar = list(itertools.combinations(range(support_rep.shape[1]), 2))
            support_et = torch.zeros(support_rep.shape[0], K, Th).float().cuda()

            for i in range(support_rep.shape[0]):
                for j in range(K):
                    support_et[i, j, :] += (support_rep[i, dar[j][0] , :] + support_rep[i, dar[j][1], :]) / 2

        support_rep = support_rep.mean(dim=1)
        local_query_rep = self.bert(input_ids=query_set, attention_mask=query_mask)[0]
        query_rep = local_query_rep[:, 0]
        logits2 = torch.zeros(query_rep.shape[0], support_et.shape[0]).float().cuda()

        if K > 1:
            for i in range(query_rep.shape[0]):
                for j in range(support_et.shape[0]):
                    for x in range(support_et.shape[1]):
                        logits2[i][j] += -((support_et[j, x, :] - query_rep[i]) ** 2).sum(dim=-1) ** 0.5
                    logits2[i][j] /= support_et.shape[1]

        support_rep = support_rep.unsqueeze(0)
        query_rep = query_rep.unsqueeze(1)
        logits = -((support_rep - query_rep) ** 2).sum(dim=-1) ** 0.5
        pred = self.ls(logits)
        if K > 1:
            pred2 = self.ls(logits2)
            res = pred * self.p + pred2 * (1.0 - self.p)
            return res, support_rep
        else:
            return pred, support_rep
