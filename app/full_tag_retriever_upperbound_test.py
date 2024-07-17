import os
import json
import dotenv
import csv
from PIL import Image

dotenv.load_dotenv()

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import TFIDFRetriever

from langchain_core.documents import Document

from datasets import load_dataset

import editdistance

import utils

document_list = []
photo_id_dict = {}
LOAD_PKL = False

TAG_CONCATENATOR = "" # ":", "-" does not work

retriever, vocab = utils.get_full_tag_retriever()

retriever.k = 100

def eval_retriever_one_query(query, img_id, top_k=10):
    retriever.k = top_k
    result = retriever.invoke(query)
    print(result[0])
    for i, found_item in enumerate(result):
        if found_item.metadata["image_path"].split("/")[-1].split(".")[0] == str(img_id):
            return i+1

    return -1

def eval_retriever(top_k=10):
    report_file = open("./tmp/tag_based_retriever_report_gt_full_tag_norm_None.csv", "w")
    report_csv_writer = csv.writer(report_file)

    eval_file = open("../fashion_test_queries_reformated_eval.csv", "r")
    eval_reader = csv.reader(eval_file)

    retriever.k = top_k
    COUNT_FOUND = 0
    COUNT_TOTAL = 0
    COUNT_BLOCKED = 0 
    for i, row in enumerate(eval_reader):
        COUNT_TOTAL += 1
        query = row[2] #row[1]

        query = " ".join([t.replace(":", TAG_CONCATENATOR) for t in query.split(",")])
        result = retriever.invoke(query)
        img_id = row[5].split("/")[-1].split(".")[0]

        print("- query:", query, "- img_id:", img_id, "- gt_tag", row[2])
        
        found_index = -1
        for j, found_item in enumerate(result):
            found_image_id = found_item.metadata["image_path"].split("/")[-1].split(".")[0]
            if found_image_id == img_id:
                COUNT_FOUND += 1
                print(f"Found top {j+1}", COUNT_FOUND, "/", COUNT_TOTAL, "BLOCKED:", COUNT_BLOCKED)
                found_index = j+1
                break
        

        if found_index == -1:
            BLOCKED = True
            print(result[0])
            for t in query.split(" "):
                if (t in result[0].page_content) == False:
                    BLOCKED = False
                    break
            if BLOCKED:
                COUNT_BLOCKED += 1

        report_csv_writer.writerow([str(img_id), str(found_index), query, row[2], ",".join(utils.tags_given_id(img_id)), str(BLOCKED)])
        report_file.flush()
        

    print("COUNT_FOUND:", COUNT_FOUND, "COUNT_TOTAL:", COUNT_TOTAL)

if __name__ == "__main__":
    eval_retriever(top_k=2000)
    import pdb
    pdb.set_trace()
