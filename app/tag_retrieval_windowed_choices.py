import dotenv

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
from langchain_core.documents import Document


from langchain_community.tools.tavily_search import TavilySearchResults
import text_to_image_retrieval
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.chains import LLMChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.runnables.base import RunnableSequence
from datasets import load_dataset



llm = ChatGroq(
        # api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0.1
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


# prompt = PromptTemplate(template = template, input_variables = ["user_query", "tag_list"])

chain = template|llm

def multiple_hop_tag_retrieval(text_query, window_size=5):
    bullet_lines = []
    for i in range(0, len(keyword_set)//2, window_size):
        print("==============> tag window:", "\n".join(keyword_set[i:i+window_size]))
        result = chain.invoke({"user_query": text_query, "tag_list":"\n".join(keyword_set[i:i+window_size])}).content
        print(result)
        for l in result.split("\n"):
            if (l.startswith("*") or l.startswith("•")) and not("None" in l):
                bullet_lines.append(l.strip())
    final_bullet_result = "\n".join([b.split(" ")[1] for b in bullet_lines])
    return final_bullet_result, bullet_lines

# print(keyword_set[:10])




raw_datasets = load_dataset(
            "simple_kfashion_query_label.py", 
            "ko", 
            cache_dir="../fashion_retriever/huggingface_datasets",
            data_dir="../fashion_retriever/data",
            split=["train", 'validation']
        )

train_dataset = raw_datasets[0]
eval_dataset = raw_datasets[1]

document_list = []


for i, item in enumerate(eval_dataset):
    page_content = ""
    for t in item['tags']:
        page_content += " " + t.replace(":", " ")
    document_list.append(Document(page_content=page_content, metadata={'image_path': item['image_path'], 'query':item['query']}))
    if i > 10:
        break
retriever = BM25Retriever.from_documents(document_list)
retriever.k = 100


import pdb
pdb.set_trace()
