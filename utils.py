import time
import os
import json
import csv
import pandas as pd
import openai
from tqdm import tqdm
import codecs


model = 'gpt-4'

def read_prompt_csv(role):
  if role == 'therapist':
     filename = 'prompts/therapist_prompts.csv'
  if role == 'client':
      filename = 'prompts/client_prompts.csv'
  df = pd.read_csv(filename)
  intent_detail_list = []
  for index, row in df.iterrows():
      positive_examples = [row['positive example 1'].strip(),row['positive example 2'].strip(), row['positive example 3'].strip()]
      negative_examples = [row['negative example 1'].strip(), row['negative example 2'].strip(), row['negative example 3'].strip()]
      intent_detail_list.append({'intent': row['intent'].strip(),'definition': row['definition'].strip(),'positive_examples': positive_examples, 'negative_examples': negative_examples})
  return intent_detail_list


def get_completion_from_messages(messages, temperature=0.7):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    time.sleep(1)
    for i in range(3):
        try:
            response = openai.ChatCompletion.create(
                                            model=model,
                                            messages=messages,
                                            temperature=temperature, 
                                        )
            return response.choices[0].message["content"]
        except:
            time.sleep(3*(2**i))
            pass
    return ''


def create_message(intent_detail_list, method, utterance, curr_intent=None):
   intent_name_list = [intent_detail['intent'] for intent_detail in intent_detail_list]
   intent_name_text = ', '.join(f'"{word}"' for word in intent_name_list)
   
   if method == 'multi_label_w_def':
      intent_definition_list = []
      for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\","")
        intent_definition_list.append(f' {intent_text}: {definition_text}')
      user_prompt_template = f"What are all possible intents of this utterance: {utterance}?\
                          Intent:\n {';'.join(intent_definition_list)}\
                          Only choose from this list: [{intent_name_text}]\
                          Please say unknown only if cannot find answer in the list. Format:[intents_list]"
      
      messages = [{'role': 'user', 'content': user_prompt_template}]
      return messages
   
   if method == 'multi_label_w_def_and_ex':
      intent_definition_with_examples_list = []
      for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\","")
        positive_example_list = intent_detail['positive_examples']

        intent_definition_with_examples_list.append(f' {intent_text}: {definition_text} Positive examples: {positive_example_list}')
      user_prompt_template = f"What are all possible intents of this therapist utterance: {utterance}?\
                                Intent:\n {';'.join(intent_definition_with_examples_list)}\
                                Only choose from this list [{intent_name_text}]\
                                Please say unknown only if cannot find answer from the list. Format:[intents_list]"
      messages = [{'role': 'user', 'content': user_prompt_template}]      
      return messages
   
   if method == 'binary_label_w_def_and_ex':
      messages = []
      for intent_detail in intent_detail_list:
        if intent_detail['intent'] != curr_intent:
          continue
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\","")
        positive_example_list = intent_detail['positive_examples']
        negative_example_list = intent_detail['negative_examples']
        system_prompt_template = f"Intent: {intent_text}\n Definition: {definition_text}\n Classify as either Yes or No."
        messages = [{'role': 'system', 'content': system_prompt_template}]

        # alternative examples
        example_label_list = []
        for example in positive_example_list[:2]:
          example_label_list.append((example,"Yes"))
        for example in negative_example_list[:2]:
          example_label_list.append((example,"No"))
        example_label_list.append((positive_example_list[2],"Yes"))
        example_label_list.append((negative_example_list[2],"No"))
        for example, label in example_label_list:
          user_utterance_template = f"Utterance: {example}"
          assistant_template = f"{label}"
          messages.append({'role': 'user', 'content': user_utterance_template})
          messages.append({'role': 'assistant', 'content': assistant_template})
        messages.append({'role': 'user', 'content': utterance})

      return messages