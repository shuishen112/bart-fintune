'''
Author: your name
Date: 2021-07-13 20:14:35
LastEditTime: 2021-07-13 20:23:29
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /bart-fintune/test_bart.py
'''
from modeling_bart import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")