import dotenv
dotenv.load_dotenv()


import csv
import json
import tqdm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_groq import ChatGroq
import time


llm = ChatGroq(
        # api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0.1
    )

user_msg = """\
- 옷 태그
{tag}

위 태그가 주어졌을 때, 이에 부합하는 구어체 한글 문장을 한 문장으로 생성해줘

"""

# user_msg = """\
# - Fashion tags
# {tag}

# Given the above tags, write a Korean sentence in spoken language that reflect the tags.
# Do not include any other output except the sentence.
# Do not use any language except Korean.

# """

template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in describing fashion style given fashion tags."),
    ("human", user_msg),
])

chain = template|llm

# Test example
TAG_EXAMPLE = "하의:카테고리:팬츠,하의:기장:맥시,하의:핏:벨보텀"
response = chain.invoke({"tag":"\n".join(TAG_EXAMPLE.split(","))})
print(response)

eval_file = open("./data/fashion_test_queries_reformated_eval.csv", "r")
eval_reader = csv.reader(eval_file)

eval_groq_file = open("./data/fashion_test_queries_reformated_eval_groq.csv", "w")
eval_groq_wrtier = csv.writer(eval_groq_file)

for i, row in tqdm.tqdm(enumerate(eval_reader)):
    tag = row[2]
    response = chain.invoke({"tag":"\n".join(tag.split(","))})
    new_row = row[:]
    new_row[1] = response.content

    eval_groq_wrtier.writerow(new_row)
    print(new_row[1])
    time.sleep(1)

eval_groq_file.close()