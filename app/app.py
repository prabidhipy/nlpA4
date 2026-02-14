import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ==========================================
# 1. BERT & S-BERT Architecture 
# ==========================================
max_len = 128
n_layers = 6
n_heads = 8
d_model = 768
d_ff = 768 * 4
d_k = d_v = 64
n_segments = 2

class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    return seq_k.data.eq(0).unsqueeze(1).expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, _ = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        return self.pos_ffn(enc_outputs)

class BERT(nn.Module):
    def __init__(self, vocab_size):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)

class SentenceBERT(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes=3):
        super(SentenceBERT, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(hidden_size * 3, num_classes)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids_a, seg_ids_a, input_ids_b, seg_ids_b):
        u_output = self.get_bert_embeddings(input_ids_a, seg_ids_a)
        v_output = self.get_bert_embeddings(input_ids_b, seg_ids_b)
        u = self.mean_pooling(u_output, (input_ids_a != 0))
        v = self.mean_pooling(v_output, (input_ids_b != 0))
        uv_abs = torch.abs(u - v)
        x = torch.cat([u, v, uv_abs], dim=-1)
        return self.classifier(x)

    def get_bert_embeddings(self, input_ids, segment_ids):
        x = self.bert.embedding(input_ids, segment_ids)
        attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.bert.layers:
            x = layer(x, attn_mask)
        return x

# ==========================================
# 2. Initialization and Model Loading
# ==========================================

# Loading custom vocabulary
try:
    with open('word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
except FileNotFoundError:
    print("ERROR: word2id.pkl not found in app directory!")
    word2id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

vocab_size = len(word2id)

# Initialize Model
base_bert = BERT(vocab_size)
model = SentenceBERT(base_bert, d_model)

model_path = 'sbert_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
model.eval()

def predict_nli(premise, hypothesis):
    def encode(text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        ids = [word2id.get(w, word2id['[MASK]']) for w in tokens][:max_len]
        return ids + [0] * (max_len - len(ids))

    ids_a = torch.LongTensor([encode(premise)])
    ids_b = torch.LongTensor([encode(hypothesis)])
    seg = torch.zeros_like(ids_a)

    with torch.no_grad():
        logits = model(ids_a, seg, ids_b, seg)
        pred = torch.argmax(logits, dim=1).item()
    
    # 0: Entailment, 1: Neutral, 2: Contradiction
    labels = ['Entailment', 'Neutral', 'Contradiction']
    return labels[pred]

# ==========================================
# 3. Routes
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    premise = data.get('premise', '')
    hypothesis = data.get('hypothesis', '')
    
    if not premise or not hypothesis:
        return jsonify({'result': 'Error: Missing inputs'})

    result = predict_nli(premise, hypothesis)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)