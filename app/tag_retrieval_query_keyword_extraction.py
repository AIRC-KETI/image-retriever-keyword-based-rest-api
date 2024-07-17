import os
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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import TFIDFRetriever

from langchain_core.documents import Document


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.chains import LLMChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.runnables.base import RunnableSequence

import editdistance

import utils
from io import BytesIO
import requests

LOAD_RETRIEVER_FROM_PKL = True

llm = ChatGroq(
        # api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0.01
    )

# llm = ChatOpenAI(
#         model="gpt-3.5-turbo",
#         temperature=0.1
#     )


# template = """\
# Given the following user query and candidate tag list,
# Select tags closely relevent to the query from the list.
# It is totally fine to print "None", if there is no relevent keyword in the list.

# USER_QUERY:
# {user_query}

# TAG_LIST:
# {tag_list}


# Answer (as a bullet list):
# """

# template = """\
# Given the following user query and candidate tag list,
# Select tags from the list that can describe the query.
# It is totally fine to print "None", if there is no relevent keyword in the list.
# You should not select tags that are not explicitly included in the query.

# USER_QUERY:
# {user_query}

# TAG_LIST:
# {tag_list}


# Answer (as a bullet list):
# """

user_msg = """\
USER_QUERY
==========
{user_query}

TAG_LIST:
==========
{tag_list}

Given the user query and candidate tag list,
Select tags from the list that are exact synonyms of words in the query.
It is totally fine to print "None", if there is no relevent keyword in the list.

Your answer should explain your match is reasonable in a pragraph, followed by a bullet list of \
(a word in the query, a word in the tag list)
==========
"""

template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in fashion style and classifying fashion vocabulary."),
    ("human", user_msg),
])


keyword_set = ['7부소매',
 'X스트랩',
 '가디건',
 '가죽',
 '골드',
 '그라데이션',
 '그래픽',
 '그레이',
 '그린',
 '글리터',
 '기장',
 '긴팔',
 '깅엄',
 '네오프렌',
 '네온',
 '네이비',
 '넥라인',
 '노멀',
 '노카라',
 '니렝스',
 '니트',
 '니트꽈베기',
 '니트웨어',
 '단추',
 '더블브레스티드',
 '데님',
 '도트',
 '드레스',
 '드롭숄더',
 '드롭웨이스트',
 '디스트로이드',
 '디테일',
 '띠',
 '라벤더',
 '라운드넥',
 '래깅스',
 '러플',
 '레드',
 '레이스',
 '레이스업',
 '레터링',
 '레트로',
 '로맨틱',
 '롤업',
 '롱',
 '루즈',
 '리본',
 '리조트',
 '린넨',
 '매니시',
 '맥시',
 '메시',
 '모던',
 '무스탕',
 '무지',
 '미니',
 '미디',
 '믹스',
 '민소매',
 '민트',
 '밀리터리',
 '반팔',
 '발목',
 '밴드칼라',
 '뱀피',
 '버클',
 '베스트',
 '베이지',
 '벨벳',
 '벨보텀',
 '보우칼라',
 '보트넥',
 '브라운',
 '브라탑',
 '브이넥',
 '블라우스',
 '블랙',
 '블루',
 '비닐/PVC',
 '비대칭',
 '비즈',
 '상의',
 '색상',
 '서브색상',
 '서브스타일',
 '세일러칼라',
 '섹시',
 '셔링',
 '셔츠',
 '셔츠칼라',
 '소매기장',
 '소재',
 '소피스트케이티드',
 '숄칼라',
 '스웨이드',
 '스위트하트',
 '스카이블루',
 '스커트',
 '스퀘어넥',
 '스키니',
 '스타일',
 '스터드',
 '스트라이프',
 '스트리트',
 '스트링',
 '스티치',
 '스판덱스',
 '스팽글',
 '스포티',
 '슬릿',
 '시퀸/글리터',
 '시폰',
 '실버',
 '실크',
 '싱글브레스티드',
 '아가일',
 '아방가르드',
 '아우터',
 '없음',
 '옐로우',
 '오렌지',
 '오리엔탈',
 '오버사이즈',
 '오프숄더',
 '옷깃',
 '와이드',
 '와인',
 '우븐',
 '울/캐시미어',
 '원숄더',
 '원피스',
 '웨스턴',
 '유넥',
 '자수',
 '자카드',
 '재킷',
 '저지',
 '점퍼',
 '점프수트',
 '젠더리스',
 '조거팬츠',
 '지그재그',
 '지브라',
 '지퍼',
 '짚업',
 '차이나칼라',
 '청바지',
 '체인',
 '체크',
 '카무플라쥬',
 '카키',
 '카테고리',
 '캡',
 '컨트리',
 '컷아웃',
 '코듀로이',
 '코트',
 '퀄팅',
 '크롭',
 '클래식',
 '키치',
 '타이다이',
 '타이트',
 '탑',
 '태슬',
 '터틀넥',
 '테일러드칼라',
 '톰보이',
 '트위드',
 '티셔츠',
 '패딩',
 '패치워크',
 '팬츠',
 '퍼',
 '퍼트리밍',
 '퍼프',
 '퍼플',
 '펑크',
 '페미닌',
 '페이즐리',
 '페플럼',
 '포켓',
 '폴로칼라',
 '폼폼',
 '프레피',
 '프린지',
 '프린트',
 '프릴',
 '플레어',
 '플로럴',
 '플리스',
 '플리츠',
 '피터팬칼라',
 '핏',
 '핑크',
 '하운즈\xa0투스',
 '하의',
 '하트',
 '하프',
 '해골',
 '헤어 니트',
 '호피',
 '홀터넥',
 '화이트',
 '후드',
 '후드티',
 '히피',
 '힙합']

chain = template|llm

document_list = []
photo_id_dict = {}

import json
# Building index...
if LOAD_RETRIEVER_FROM_PKL:
    retriever = TFIDFRetriever.load_local("retriever.pkl", allow_dangerous_deserialization=True)
else:
    kfashion_images_group_reader = csv.reader(open("../kfashion_images_group.tsv", "r"), delimiter="\t")
    for i, row in enumerate(kfashion_images_group_reader):

        img_id = row[1].split("/")[-1].split(".")[0]
        tag_json = json.load(open(f"../fashion_retriever/data/k_fashion_labels/{img_id}.json", "r"))

        img_labelings = tag_json['데이터셋 정보']['데이터셋 상세설명']['라벨링']

        img_labeling_set = set()
        colon_concated_tags = []

        for k, v in img_labelings.items():
            img_labeling_set.add(k)
            if isinstance(v, list):
                for d in v:
                    for kk, vv in d.items():
                        img_labeling_set.add(kk)
                        if isinstance(vv, list):
                            for vvv in vv:
                                img_labeling_set.add(vvv)
                                colon_concated_tags.append(":".join([k, kk, vvv]))
                        else:
                            img_labeling_set.add(vv)
                            colon_concated_tags.append(":".join([k, kk, vv]))
            else:
                img_labeling_set.add(v)


        page_content = " ".join(list(img_labeling_set))
        
        document_list.append(Document(page_content=page_content, metadata={'image_path': row[1], 'query':"EMPTY for compatibility", 'image_tag_set':",".join(colon_concated_tags)}))
        

        photo_id_dict[img_id] = document_list[-1]
        if i % 100 == 0:
            print(i)

        # if len(document_list) > 100:
        #     break
        
    # retriever = BM25Retriever.from_documents(document_list)
    retriever = TFIDFRetriever.from_documents(document_list)
    retriever.save_local("retriever.pkl")

retriever.k = 100

def extract_keywords(query):
    response = llm.invoke(f"Extract fashion-related atomic keywords from the user query and make a bullet list.\
                          Each keyword should be in Korean and do not include english translations: {query}")
    print(response.content)
    keyword_list = []
    for l in response.content.split("\n"):
        if l.startswith("•"):
            # exclude "•" and following english keyword in paranthesis. 
            # keyword_list.append(l[1:].strip().split(" ")[0]) 
            
            # exclude "•" and following english keyword in paranthesis. 
            # remove tailing period.
            keyword_list.append(l[1:].strip().split("(")[0].split(".")[0]) 

    return keyword_list

def find_the_best_match(query_tag):
    if query_tag in keyword_set:
        return query_tag
        
    response = llm.invoke(\
"""Find the best match of "{query_tag}" in the following candidate tag list:
TAG_LIST
========
{keyword_set}

Your Answer 
- Find the most semanically similar match.
- The match should be contained in the above tag list.
- End your answer with a line in the following format:
    Found tag: <found tag in the TAG_LIST>.
- If no match is found, print "None".
========
"""\
.format(query_tag=query_tag, keyword_set="\n".join(keyword_set)))
    print(response.content)
    # remove tailing period.
    answered_tag = response.content.split("Found tag:")[1].strip().split(".")[0]
    correct_tag = answered_tag
    min_editdistance = 9999999

    if not(answered_tag in keyword_set) and answered_tag.lower() != "none":
        for t in keyword_set:
            test_edit_distance = editdistance.eval(t, answered_tag)
            if test_edit_distance < min_editdistance:
                min_editdistance = test_edit_distance
                correct_tag = t
            if test_edit_distance == 0:
                # Early stopping of test.
                break


    return correct_tag 

def query_to_keywords(query):
    best_match_pair_list = [(kw, find_the_best_match(kw)) for kw in extract_keywords(query)]
    return best_match_pair_list

def eval_retriever_one_query(query, img_id, top_k=10):
    retriever.k = top_k
    query_tags_to_search_tags = query_to_keywords(query)
    search_keyword = " ".join([sk for (qk, sk) in query_tags_to_search_tags])
    print("search_keyword:", search_keyword, "query_tags_to_search_tags:", query_tags_to_search_tags)
    result = retriever.invoke(search_keyword)
    print(result[0])
    for i, found_item in enumerate(result):
        if found_item.metadata["image_path"].split("/")[-1].split(".")[0] == str(img_id):
            return i+1

    return -1

def eval_retriever(top_k=10):
    report_file = open("./tmp/tag_based_retriever_report_gt.csv", "w")
    report_csv_writer = csv.writer(report_file)

    eval_file = open("../fashion_test_queries_reformated_eval.csv", "r")
    eval_reader = csv.reader(eval_file)

    retriever.k = top_k
    COUNT_FOUND = 0
    COUNT_TOTAL = 0
    for i, row in enumerate(eval_reader):
        query = row[2] #row[1]

        search_keyword = " ".join([sk for (qk, sk) in query_to_keywords(query)])
        print("search_keyword:", )
        result = retriever.invoke(search_keyword)
        img_id = row[5].split("/")[-1].split(".")[0]

        print("- query:", query, "- search:", search_keyword, "- img_id:", img_id, "- gt_tag", row[2])
        
        found_index = -1
        for j, found_item in enumerate(result):
            found_image_id = found_item.metadata["image_path"].split("/")[-1].split(".")[0]
            if found_image_id == img_id:
                COUNT_FOUND += 1
                print(f"Found top {j+1}", COUNT_FOUND, "/", COUNT_TOTAL)
                found_index = j+1
                break
        report_csv_writer.writerow([str(img_id), str(found_index), query, search_keyword, row[2], ",".join(utils.tags_given_id(img_id))])
        report_file.flush()
        COUNT_TOTAL += 1
        
        
    print("COUNT_FOUND:", COUNT_FOUND, "COUNT_TOTAL:", COUNT_TOTAL)


        

def find_image(txt, return_found_tags=False, return_image_as_url=False, return_bucket_url=False, num_images=30):
    detected_tags = query_to_keywords(txt)
    detected_tags_concat = ", ".join([v[1] for v in detected_tags])
    
    old_top_k = retriever.k
    retriever.k = num_images
    result = retriever.invoke(detected_tags_concat)
    retriever.k = old_top_k

    found_image_tags_list = []
    
    print("is pressed?!")
    print(result)
    print(detected_tags)
    image_list = []
    local_path_root = "/mnt/nfs4/byunggill/multimodal/k_fashion/fashion_retriever/data/k_fashion_images/"
    # bucket_url = f"https://storage.googleapis.com/k_fashion_images/k_fashion_images/{img_id}.jpg"
    for i in range(num_images):
        image_list.append(utils.get_url_output(result[i].metadata['image_path'], return_image_as_url, return_bucket_url))
        found_image_tags_list.append(result[i].metadata['image_tag_set'])

    print("found_image_tags_list:", found_image_tags_list)
    if return_found_tags:
        return image_list, detected_tags_concat, found_image_tags_list
    else:
        return image_list, detected_tags_concat

if __name__ == "__main__":
    eval_retriever(top_k=2000)
    import pdb
    pdb.set_trace()
