from IR_transformers.modeling_t5 import T5ForConditionalGeneration
from IR_transformers.tokenization_t5 import T5Tokenizer
import torch
import logging 
model = T5ForConditionalGeneration.from_pretrained("/data/ceph/zhansu/embedding/t5-small")
tokenizer = T5Tokenizer.from_pretrained("/data/ceph/zhansu/embedding/t5-small")


input_ids = tokenizer('how are glacier caves formed ?', return_tensors='pt').input_ids
labels = tokenizer('A glacier cave is a cave formed within the ice of a glacier .', return_tensors='pt').input_ids
# the forward function automatically creates the correct decoder_input_ids

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level = logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

epoch = 100

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

for i in range(epoch):
    loss = model(input_ids=input_ids, labels=labels).loss

    # 计算反向传播的值
    loss.backward()

    # 反向传播更新梯度
    optimizer.step()

    optimizer.zero_grad()
    logger.info("train loss:{}".format(loss.item()))
    print("train loss",loss.item())

#训练完成后进行生成句子

# beam search
beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))



# print("T5 loss",loss)

# from IR_transformers.modeling_bart import BartForConditionalGeneration
# from IR_transformers.tokenization_bart import BartTokenizer

# model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# loss_bart = model_bart(input_ids=input_ids,labels = labels).loss
# print("Bart loss",loss_bart)



