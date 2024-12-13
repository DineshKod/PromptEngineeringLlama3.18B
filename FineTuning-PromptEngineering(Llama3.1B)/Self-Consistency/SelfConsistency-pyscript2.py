# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")

# Use a pipeline as a high-level helper
from transformers import pipeline

import pandas as pd
from collections import Counter

dfr = pd.read_csv('Questions3.csv', encoding='utf-8')
print(dfr)
for index, row in dfr.iterrows():
    print(index)
    body = row['question']
    temperature = 0.7
    prompt = "Please answer this question: " + body + " Show all work. Output your answer like this: \#\#\#work: <work> \#\#\# Answer: <answer>."
    
    for i in range(3):
        messages = [
            {"role": "user", "content": prompt},
            ]
        pipe = pipeline("text-generation", model="NousResearch/Hermes-3-Llama-3.1-8B", temperature=temperature, tokenizer=tokenizer, max_new_tokens=700)
        pipe(messages)
        outputs = pipe(messages, max_new_tokens=700,)
        print(outputs)
        output_text = outputs[0]["generated_text"][-1]['content']
        print(output_text)
        dfr.loc[index, 'Prompt'] = prompt
        dfr.loc[index, 'Response' + str(i)] = output_text
print('done1')
dfr.to_csv('out_selfconsistency3.csv')
dfr['Answer0'] = dfr['Response0'].str.split('Answer:').str[1]
dfr['Answer1'] = dfr['Response1'].str.split('Answer:').str[1]
dfr['Answer2'] = dfr['Response2'].str.split('Answer:').str[1]
df = pd.read_csv('out_selfconsistency3.csv',  encoding='utf-8')
print(df)

for index2, row2 in df.iterrows():
    print(index2)
    response0 = row2['Response0']
    response1 = row2['Response1']
    response2 = row2['Response2']
    answers = response0, response1, response2
    most_freq_ans, frequency = Counter(answers).most_common(1)[0]
    df.loc[index2, 'Most frequent'] = most_freq_ans
    df.loc[index2, 'Frequency'] = frequency

print('done2')

df.to_csv('out_selfconsistency3.csv')

print('done3')
