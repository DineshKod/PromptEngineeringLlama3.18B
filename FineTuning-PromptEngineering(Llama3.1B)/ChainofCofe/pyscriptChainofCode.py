# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")

# Use a pipeline as a high-level helper
from transformers import pipeline

import pandas as pd

df = pd.read_csv('Questions.csv',  encoding='utf-8')
print(df)

for index, row in df.iterrows():
    print(index)
    body = row['question']
    prompt = 'Please write a python code that will output the answer to this question: ' + body + ' Only write python code. Store the output in variable Ans in your python code.'
    messages = [
                {"role": "user", "content": prompt},
                ]
    pipe = pipeline("text-generation", model="NousResearch/Hermes-3-Llama-3.1-8B", tokenizer=tokenizer, max_new_tokens=500)
    pipe(messages)
    outputs = pipe(messages, max_new_tokens=500,)
    print(outputs)
    output_text = outputs[0]["generated_text"][-1]['content']
    print(output_text)
    df.loc[index, 'Prompt'] = prompt
    df.loc[index, 'Response'] = output_text
print('done')
df.to_csv('AnswersCoC3.csv')

df2 = pd.read_csv('AnswersCoC3.csv', encoding='utf-8')
for index2, row2 in df2.iterrows():
    print(index2)
    try:
        code = row2['Response']
        exec(code)
        df2.loc[index2, 'Answer'] = Ans
    except Exception as e:
        print(f"Error at index {index2}: {e}")
        df2.loc[index2, 'Answer'] = None

df2.to_csv('AnswersCoC3.csv')

print('done 2')
