# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")

# Use a pipeline as a high-level helper
from transformers import pipeline

import pandas as pd

dfr = pd.read_csv('RAGq.csv', encoding='utf-8')
print(dfr)
rag1 = dfr.at[0, 'question']
rag2 = dfr.at[1, 'question']
rag3 = dfr.at[2, 'question']
rag4 = dfr.at[3, 'question']
rag5 = dfr.at[4, 'question']
print(rag1, rag2, rag3, rag4, rag5) 
for index, row in dfr.iterrows():
    print(index)
    question = row['question']
    prompt = "I will share some data and a question with you. Read the data and answer the question. If required use and refer to the data to answer the question. Data: " + rag1 + rag2 + rag3 + rag4 + rag5 + " Question: " + question + " Show all work and if you used the data or not. Output your answer like this: \#\#\#work: <work> \#\#\# Answer: <answer>"
    messages = [
            {"role": "user", "content": prompt},
            ]
    pipe = pipeline("text-generation", model="NousResearch/Hermes-3-Llama-3.1-8B", tokenizer=tokenizer, max_new_tokens=1000)
    pipe(messages)
    outputs = pipe(messages, max_new_tokens=1000,)
    print(outputs)
    output_text = outputs[0]["generated_text"][-1]['content']
    print(output_text)
    dfr.loc[index, 'Prompt'] = prompt
    dfr.loc[index, 'Response'] = output_text
print('done1')

dfr.to_csv('RAG_output.csv')

print('done3')
