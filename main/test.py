import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model

input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
print(input_ids)
print(input_mask)

token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
model, vocab = get_pytorch_kobert_model()
print(token_type_ids)
print(model)
print(vocab)

output = model(input_ids, input_mask, token_type_ids)
print(output.pooler_output.shape) # Use `output` instead of using `pooled_output`.
print(output[0]) # Use `output` instead of using `sequence_output[0]`.
