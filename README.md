Aim: Test various prompting strategies on Llama 3.18B. 

Database: Questions.csv contains 112 7th grade mathematics questions and it's answers. These questions are prompted to the LLM following various techniques. In special cases like few shot and RAG, we retrieve data from outside this file. 

Methodology: 
We test in a total of 5 prompting strategies, as follows:
0. Zero-Shot (This acts as base for our study) 
1. Chain of Thought
2. Few Shot
3. Chain of Code
4. Retrival Augmented Generation (RAG)
5. Self-Consistency

Results: 
Chain of Thought outperformed all other prompting strategies, which had a statistically significant difference with Zero shot.
![image](https://github.com/user-attachments/assets/99aca230-50fc-4ac4-9063-9a44fe2bebac)

