import time
import os
import json
import csv
import argparse
import pandas as pd
import openai
from tqdm import tqdm
import codecs

from utils import *

intent_detail_list = read_prompt_csv('therapist')

def get_therapist_intent(utterance, method='multi_label_w_def_and_ex'):
	utterance = utterance.strip()
     
	messages = create_message(intent_detail_list, method, utterance)
	
	response = get_completion_from_messages(messages, temperature=0.7)
	
	return response


parser = argparse.ArgumentParser(description='Behavior Inference')

parser.add_argument('--method', type=str, default='multi_label_w_def_and_ex', help='method', choices=['multi_label_w_def_and_ex', 'binary_label_w_def_and_ex', 'multi_label_w_def',])
parser.add_argument('--input_path', type=str, default='dataset/sample_therapist_input.jsonl', help='Path to input')
parser.add_argument('--output_path', type=str, default='dataset/sample_therapist_output.jsonl', help='Path to output')

args = parser.parse_args()

method = args.method
input_path = args.input_path
output_path = args.output_path

print('Method: ', method)
print('Input Path: ', input_path)
print('Output Path: ', output_path)

	
f = codecs.open(input_path, 'r', 'utf-8')
output_f = codecs.open(output_path, 'w', 'utf-8')

for row in tqdm(f):
	curr_json = json.loads(row.strip())

	curr_json['therapist_behavior'] = get_therapist_intent(curr_json['utterance'], method=method)
	
	print(json.dumps(curr_json), file=output_f)

f.close()
output_f.close()