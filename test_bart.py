'''
Author: your name
Date: 2021-07-13 20:14:35
LastEditTime: 2021-07-13 22:08:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /bart-fintune/test_bart.py
'''
from modeling_bart import BartForConditionalGeneration
from tokenization_bart import BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def query_expansion(query):
    querys = []
    inputs = tokenizer([query], max_length=1024, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], num_return_sequences = 5, num_beams=5, max_length=50, early_stopping=True,output_scores = True,return_dict_in_generate =True)
    sequences = outputs['sequences']
    sequences_scores = outputs['sequences_scores']
    for s in sequences:
        querys.append(tokenizer.decode(s, skip_special_tokens = True, clean_up_tokenization_spaces = False))
    return (querys,sequences_scores)


ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
query_output = query_expansion(ARTICLE_TO_SUMMARIZE)

for query,score in zip(*query_output):
    print(query,score)
