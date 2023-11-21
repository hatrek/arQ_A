import pandas as pd
import torch
from transformers import ElectraForQuestionAnswering, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



df = pd.read_json('dataset.json')
cols = ["text","question","answer"]
comp_list = []
for index, row in df.iterrows():
    for i in range(len(row["data"]["questions"])):
        temp_list = []
        temp_list.append(row["data"]["story"])
        temp_list.append(row["data"]["questions"][i]["text"])
        temp_list.append(row["data"]["answers"][i])
       
        comp_list.append(temp_list)
new_df = pd.DataFrame(comp_list, columns=cols)


def setupData(data):
  contexts = []
  questions = []
  answers = []

  for index, stor in new_df.iterrows():
      contexts.append(stor['text'])
  for rows in data['data']:
    for qa in rows['questions']:
      questions.append(qa['text'])
    for answer in rows['answers']:
        print(len(answer['text']))
        answers.append(answer)
  return contexts, questions, answers


data = pd.read_json("dataset.json")
train_contexts, train_questions, train_answers = setupData(data)
valid_contexts, valid_questions, valid_answers = setupData(data)



modelName = 'model_files/'
model = ElectraForQuestionAnswering.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)


string = ' '.join(train_contexts)
vocab = string.split()
token_vocab = list(set(vocab))
new_tokens = set(token_vocab) - set(tokenizer.vocab.keys())


tokenizer.add_tokens(list(new_tokens))
model.resize_token_embeddings(len(tokenizer))


# حفظ tokenizer
save_path = "tokneizer/"
tokenizer.save_pretrained(save_path)
newtokenizer = AutoTokenizer.from_pretrained('tokneizer/')



def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    answer['answer_start'] = context.find(answer['text'])
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
      answer['answer_start'] = start_idx - 1
      answer['answer_end'] = end_idx - 1 
    elif context[start_idx-2:end_idx-2] == gold_text:
      answer['answer_start'] = start_idx - 2
      answer['answer_end'] = end_idx - 2  

add_end_idx(train_answers, train_contexts)
add_end_idx(valid_answers, valid_contexts)




train_encodings = newtokenizer(train_contexts, train_questions,truncation=True, max_length=512,padding='longest')
valid_encodings = newtokenizer(valid_contexts, valid_questions,truncation=True,max_length=512, padding='longest')


def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
    if start_positions[-1] is None:
      start_positions[-1] = newtokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = newtokenizer.model_max_length
  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(valid_encodings, valid_answers)




class MY_Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)
  
train_dataset = MY_Dataset(train_encodings)
valid_dataset = MY_Dataset(valid_encodings)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

N_EPOCHS = 30
optim = AdamW(model.parameters(), lr=0.001)
model.to(device)
model.train()
for epoch in range(N_EPOCHS):
  loop = tqdm(train_loader, leave=True)
  for batch in loop:
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
    loss = outputs[0]
    loss.backward()
    optim.step()

    loop.set_description(f'Epoch {epoch+1}')
    loop.set_postfix(loss=loss.item())



model_path = 'mymodel/'
model.save_pretrained(model_path)
newtokenizer.save_pretrained(model_path)