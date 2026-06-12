from functools import partial
from hypothesis.LoRA import Linear_With_LoRA

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
for param in model.parameters():
    param.requires_grad = False

print(model)

lora_rank = 8
lora_alpha = 15
assign_lora = partial(Linear_With_LoRA, rank=lora_rank, alpha=lora_alpha, LoRa_type='stander')

# 改QKV 和 FFN
for layer in model.distilbert.transformer.layer:
    layer.attention.q_lin = assign_lora(layer.attention.q_lin)
    layer.attention.k_lin = assign_lora(layer.attention.k_lin)
    layer.attention.v_lin = assign_lora(layer.attention.v_lin)

    layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
    layer.ffn.lin2 = assign_lora(layer.ffn.lin2)

# 改下游头
model.classifier = assign_lora(model.classifier)

print(model)