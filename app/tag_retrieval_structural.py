import dotenv
dotenv.load_dotenv()

import os
import csv
import json
import tqdm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_groq import ChatGroq
import time
import utils
import pickle
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

llm = ChatGroq(
        # api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0.1
    )


def load_dataset(configs):
    query_file_path = configs["query_file_path"]
    eval_groq_file = open(query_file_path, "r")
    eval_groq_reader = csv.reader(eval_groq_file)
    elements = []
    
    for i, row in enumerate(eval_groq_reader):
        elements.append(row)
    
    return elements


def sentence_to_category(description):

    user_msg = """\
    - 옷 설명
    {description}

    옷 설명이 주어졌을 때 다음 중 어떤 카테고리의 옷에 대한 설명인지 번호만 선택해서 알려줘.
    여러 번호에 해당할 경우 "," 로 연결해서 알려줘.
    
    1. 상의
    2. 하의
    3. 원피스
    4. 아우터

    """
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in classifying fashion description to a related fashion category."),
        ("human", user_msg),
    ])

    chain = template|llm

    response = chain.invoke({"description":description})

    return response.content


def test_sentence_to_category(configs):
    dataset = load_dataset(configs)
    # import pdb
    # pdb.set_trace()
    TOTAL_CNT = 0
    CORRECT_CNT = 0
    category_tag_list = ["상의", "하의", "원피스", "아우터"]

    for i, r in enumerate(dataset):
        
        output = sentence_to_category(r[1])
        print("[**]", r[1], "::", output)
        output_index = int(output.strip().split(",")[0])
        assert(output_index > 0 and output_index < 5)
        TOTAL_CNT += 1
        if (category_tag_list[output_index - 1] + ":") in r[2]:
            CORRECT_CNT += 1

        print(CORRECT_CNT, "/", TOTAL_CNT)


def load_category_to_tag_set_dict(configs):
    USE_CACHE = configs["use_category_to_tags_cache"]
    INDEX_FILE_PATH = configs["index_file_path"]
    CACHE_DIR_PATH = configs["category_to_tags_cache_dir_path"]

    CACHE_PATH1 = os.path.join(CACHE_DIR_PATH, "category_tag_set_dict.pickle")
    CACHE_PATH2 = os.path.join(CACHE_DIR_PATH, "other_tag_set.pickle")

    if USE_CACHE and os.path.exists(CACHE_PATH1) and os.path.exists(CACHE_PATH2):
        FILE1, FILE2 = open(CACHE_PATH1, "rb"), open(CACHE_PATH2, "rb")
        category_tag_set_dict = pickle.load(FILE1)
        other_tag_set = pickle.load(FILE2)
        FILE1.close()
        FILE2.close()

        return category_tag_set_dict, other_tag_set


    index_tsv_file = open(INDEX_FILE_PATH, "r")
    index_tsv_reader = csv.reader(index_tsv_file, delimiter="\t")
    category_tag_set_dict = {"상의":set(), "하의":set(), "원피스":set(), "아우터":set()}
    other_tag_set = set()
    for i, r in enumerate(index_tsv_reader):
        img_id = int(r[0].split(".")[0])
        img_tags = utils.tags_given_id(img_id)
        for t in img_tags:
            ADDED = False
            for k in category_tag_set_dict.keys():
                if k+":" in t:
                    category_tag_set_dict[k].add(t)
                    ADDED = True
                    break

            if ADDED == False:
                other_tag_set.add(t)

    with open(CACHE_PATH1, "wb") as fw:
        pickle.dump(category_tag_set_dict, fw)

    with open(CACHE_PATH2, "wb") as fw:
        pickle.dump(other_tag_set, fw)

    return category_tag_set_dict, other_tag_set


def sentence_to_tag(description, tags):
    # 해당하는 태그의 번호를 '*'로 시작하는 bullet list로 출력해줘.

    user_msg = """\
- 옷 설명
{description}

옷 설명이 주어졌을 때 옷을 꾸미는 말들을 뽑고,
각 꾸미는 말에 부합하는 태그의 번호를 아래에서 선택해줘. "* <번호>. <태그> " 형식이야.
애매하게 부합하는 태그는 안되고, direct match 태그들만 선택해야해.


{tag}

"""
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in classifying fashion description to a related fashion category."),
        ("human", user_msg),
    ])

    chain = template|llm
    tag_str = ""
    for i, t in enumerate(tags):
        tag_str += f"{i+1}. {t}\n"
    
    # response = chain.invoke({"description":description, "tag":"\n".join(tag)})
    response = chain.invoke({"description":description, "tag":tag_str})
    print(tag_str)
    print(response.content)
    indice = [int(r.split(".")[0]) - 1 for r in response.content.split("*")[1:]]
    print(indice)
    return response.content, indice


def test_sentence_to_tag(configs):
    dataset = load_dataset(configs)
    # import pdb
    # pdb.set_trace()
    category_tag_list = ["상의", "하의", "원피스", "아우터"]
    
    category_tag_set_dict, other_tag_set = load_category_to_tag_set_dict(configs)
    
    for k in category_tag_set_dict.keys():
        category_tag_set_dict[k] = list(category_tag_set_dict[k])

    COUNT_TRUE_POSITIVE_TAG = 0
    COUNT_POSITIVE_TAG = 0
    COUNT_GT_TAG = 0

    INSTANCE_RECALL = 0
    INSTANCE_PRECISION = 0 

    for i, r in enumerate(dataset):
        while True:
            try:
                output = sentence_to_category(r[1])
                output_index = int(output.strip().split(",")[0])
                assert(output_index > 0 and output_index < 5)

                category = category_tag_list[output_index - 1]
                
                response, tag_indice = sentence_to_tag(r[1], category_tag_set_dict[category])
                # make unique tag list
                tag_list = list(set([category_tag_set_dict[category][ti] for ti in tag_indice]))
                gt_tag_list = [gt.strip() for gt in r[2].split(",")]

                TRUE_POSITIVE = len([t for t in tag_list if t in gt_tag_list])
                POSITIVE = len(tag_list)
                GT = len(gt_tag_list)

                INSTANCE_RECALL += TRUE_POSITIVE/GT
                INSTANCE_PRECISION += TRUE_POSITIVE/POSITIVE

                COUNT_TRUE_POSITIVE_TAG += TRUE_POSITIVE
                COUNT_POSITIVE_TAG += POSITIVE
                COUNT_GT_TAG += GT
                time.sleep(1)

                print(i, category, tag_list, gt_tag_list)
                print("INSTANCE_RECALL", INSTANCE_RECALL/(i+1),\
                        "INSTANCE_PRECISION", INSTANCE_PRECISION/(i+1),\
                        "GLOBAL_RECALL", COUNT_TRUE_POSITIVE_TAG / COUNT_GT_TAG, \
                        "GLOBAL_PRECISION", COUNT_TRUE_POSITIVE_TAG / COUNT_POSITIVE_TAG)
            except:
                continue
            else:
                break

    print("INSTANCE_RECALL", INSTANCE_RECALL/len(dataset),\
          "INSTANCE_PRECISION", INSTANCE_PRECISION/len(dataset),\
          "GLOBAL_RECALL", COUNT_TRUE_POSITIVE_TAG / COUNT_GT_TAG, \
          "GLOBAL_PRECISION", COUNT_TRUE_POSITIVE_TAG / COUNT_POSITIVE_TAG)


def decide_style_necessity_v1(description, tags):
    user_msg = """\
- 옷 설명
{description}

- 옷 설명 태그
{tag}

"옷 설명"이 있고, 옷 설명에 부합하는 "옷 설명 태그"가 있는데
현재 "옷 설명 태그"가 "옷 설명"을 나타내는데 충분하니?

충분하다면 "예"를 출력하고, 구체적 스타일을 나타내는 태그가 더 필요하다면 "아니오"를 출력해.


"""
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in checking whether fashion tags match a natural language description of fashion."),
        ("human", user_msg),
    ])

    chain = template|llm

    response = chain.invoke({"description":description, "tag":"\n".join(tags)})

    return response.content

def decide_style_necessity_v2(description, tags, style_tags):
    """


    "Fashion description"과 "List1"의 태그들을 비교한 후,
    만약 "Fashion description"이 "List1"에 포함되지 않았지만
    "List2"에 포함 된 스타일 속성을 포함한다면 "아니오"를 출력하시오.
    그렇지 않다면 "예"를 출력하시오.
    """

    user_msg = """\
- Fashion description
{description}

- List2 (Fashion description을 더 잘 설명하기 위해 사용 가능한 태그 목록)
{style_tag}

당신은 Fashion style expert 입니다. 미묘한 패션 차이까지 잘 잡아냅니다.

1. "Fashion description"에 포함된 스타일 키워드를 추출하시오. 
2. 각 키워드에 정확히 부합하는 태그가 List2에 있다면 출력하시오. 
번호까지 출력하시오. 그렇지 않다면 절대 출력하지 마시오. 정확히 부합하지 않고 유사하기만 하다면 태그 대신 None을 출력하시오.
아래의 output format 에 맞도록 출력하고, 다른 내용은 출력하지 마시오.

- Output format
<a bullet list of keywords extracted from "Fashion description">

<a bullet list of exact-match tags in List2 corresponding to the above bullet list>

"""
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in checking whether fashion tags match a natural language description of fashion."),
        ("human", user_msg),
    ])

    chain = template|llm

    style_tags_str = ""
    for i, t in enumerate(style_tags):
        style_tags_str += f"{i}. {style_tags[i]}\n"

    response = chain.invoke({"description":description, "tag":"\n".join(tags), "style_tag":style_tags_str})

    return response.content

def test_decide_style_necessity(configs):
    # Given extracted tags and natural language description,
    # Test decision accuracy whether it needs style tags.

    dataset = load_dataset(configs)
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 0
    TRUE_NEGATIVE = 0
    FALSE_NEGATIVE = 0
    category_tag_set_dict, other_tag_set = load_category_to_tag_set_dict(configs)
    other_tag_set = list(other_tag_set)
    other_tag_set_last = [t.split(":")[-1] for t in other_tag_set]

    for i, r in enumerate(dataset):
        description = r[1].strip()
        tags = [gt.strip() for gt in r[2].split(",")]
        tags_wo_style = [t for t in tags if t.startswith("스타일:") == False]

        IS_INCLUDE_STYLE = len(tags_wo_style) != len(tags)
        # if IS_INCLUDE_STYLE == False:
        #     continue
        output = decide_style_necessity_v2(description, tags_wo_style, other_tag_set)
        
        is_negative = "아니오" in output


        if is_negative and IS_INCLUDE_STYLE == False:
            TRUE_NEGATIVE += 1
        elif is_negative and IS_INCLUDE_STYLE == True:
            FALSE_NEGATIVE += 1
        elif is_negative == False and IS_INCLUDE_STYLE == True:
            TRUE_POSITIVE += 1
        elif is_negative == False and IS_INCLUDE_STYLE == False:
            FALSE_POSITIVE += 1
        
        print("-----")
        print("[*]", description, tags, output)
        print("* Confusion matric", "\n", TRUE_POSITIVE, FALSE_NEGATIVE,"\n", FALSE_POSITIVE, TRUE_NEGATIVE)


def retrieve_using_kernel(retriever, query, configs):
    kernel = None
    if configs["retrieval_kernel"] == "linear":
        kernel = linear_kernel
    elif configs["retrieval_kernel"] == "cosine_similarity":
        kernel = cosine_similarity
    
    
    query_vec = retriever.vectorizer.transform(
        [query]
    )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
    
    results = kernel(retriever.tfidf_array, query_vec).reshape(
        (-1,)
    )  # Op -- (n_docs,1) -- Cosine Sim with each doc

    
    return_docs = [retriever.docs[i] for i in results.argsort()[-retriever.k :][::-1]]
    
    return return_docs


def test_structural_retrieval(configs):
    # retriever
    retriever, vocab = utils.get_full_tag_retriever(configs)
    
    # The following retriever does not make sense
    # retriever, vocab = utils.get_full_tag_retriever_refine_eval_data_tags(configs)
    
    retriever.k = configs["k"]
    # search eval data
    dataset = load_dataset(configs)

    # category data
    category_tag_list = ["상의", "하의", "원피스", "아우터"]
    category_tag_set_dict, other_tag_set = load_category_to_tag_set_dict(configs)
    
    for k in category_tag_set_dict.keys():
        category_tag_set_dict[k] = list(category_tag_set_dict[k])
    other_tag_set = list(other_tag_set)

    COUNT_FOUND = 0
    COUNT_TOTAL = 0
    COUNT_BLOCKED_STRONG = 0
    COUNT_BLOCKED_WEAK = 0
    for i, r in enumerate(dataset):
        COUNT_TOTAL += 1

        while True:
            try:
                # gt_category_list = set([t.split(":")[0] for t in r[2].split(",") if any([t.startswith("상의:"), t.startswith("하의:"), t.startswith("원피스:"), t.startswith("아우터:")])])
                # if len(gt_category_list) <= 1:
                #     print(gt_category_list, "continue")
                #     break

                output = sentence_to_category(r[1])
                output_indice = list(map(int, output.strip().split(",")))
                # print(gt_category_list, output_indice, output)

                tag_list = []
                for output_index in output_indice:
                    assert(output_index > 0 and output_index < 5)

                    category = category_tag_list[output_index - 1]
                    
                    response, tag_indice = sentence_to_tag(r[1], category_tag_set_dict[category] + other_tag_set)
                    # make unique tag list
                    CATEGORY_TAG_LEN = len(category_tag_set_dict[category])
                    category_tag_indice = [ti for ti in tag_indice if ti < CATEGORY_TAG_LEN]
                    style_tag_indice = [ti - CATEGORY_TAG_LEN for ti in tag_indice if ti >= CATEGORY_TAG_LEN]

                    tag_list = tag_list + list(set([category_tag_set_dict[category][ti] for ti in category_tag_indice]))\
                                + list(set([other_tag_set[ti] for ti in style_tag_indice]))

                gt_tag_list = [gt.strip() for gt in r[2].split(",")]
                tag_wo_colon_list = ["".join(t.split(":")) for t in tag_list]
                
                # It is not necessary to check "t in vocab.keys()", because all tags is from 
                # all_category_tags and other_tag_set
                # for t in tag_wo_colon_list:
                #     assert(t in vocab.keys())
                
                search_tag_query = " ".join(tag_wo_colon_list)
                retrieval_result = retrieve_using_kernel(retriever, search_tag_query, configs)
                # retrieval_result = retriever._get_relevant_documents(search_tag_query, run_manager=None)
                # retrieval_result = retrieve_using_dot(retriever, search_tag_query) 
            
                img_id = r[5].split("/")[-1].split(".")[0].strip()

                time.sleep(1)


                found_index = -1
                for j, found_item in enumerate(retrieval_result):
                    found_image_id = found_item.metadata["image_path"].split("/")[-1].split(".")[0].strip()
                    # print(found_image_id, img_id)
                    if found_image_id == img_id:
                        COUNT_FOUND += 1
                        found_index = j+1
                        break

                
                BLOCKED_STRONG = False
                BLOCKED_WEAK = False
                if found_index == -1:
                    BLOCKED_STRONG = True
                    BLOCKED_WEAK = True
                    
                    # Regardless of correctness in tag extraction, if top 1 result contains ground truth tags
                    # It is BLOCKED_WEAK
                    # For compatibility with dense retriever.
                    for t in gt_tag_list:
                        if ("".join(t.split(":")) in retrieval_result[0].page_content) == False:
                            BLOCKED_WEAK = False
                            break

                    # if tag extraction is correct "and" top 1 result contains ground truth tags
                    # It is BLOCKED_STRONG

                    for t in tag_wo_colon_list:
                        if (t in retrieval_result[0].page_content) == False:
                            BLOCKED_STRONG = False
                            break
                    
                    if len(tag_list) != len(gt_tag_list):
                        BLOCKED_STRONG = False
                    else:
                        for t in tag_list:
                            if (t in gt_tag_list) == False:
                                BLOCKED_STRONG = False
                                break
                        
                    if BLOCKED_WEAK:
                        COUNT_BLOCKED_WEAK += 1

                    if BLOCKED_STRONG:
                        COUNT_BLOCKED_STRONG += 1

                    
                print(f"Found top {found_index}", COUNT_FOUND, "/", COUNT_TOTAL, COUNT_BLOCKED_STRONG, COUNT_BLOCKED_WEAK)
                print(i, found_index, category, tag_list, gt_tag_list, BLOCKED_STRONG, BLOCKED_WEAK)
                
            except:
                print("An error occured")
                continue
            else:
                break


def test_flat_retrieval(configs):
    # retriever
    retriever, vocab = utils.get_full_tag_retriever(configs)
    
    # The following retriever does not make sense
    # retriever, vocab = utils.get_full_tag_retriever_refine_eval_data_tags()
    
    retriever.k = configs["k"]
    # search eval data
    dataset = load_dataset(configs)

    # category data
    category_tag_list = ["상의", "하의", "원피스", "아우터"]
    category_tag_set_dict, other_tag_set = load_category_to_tag_set_dict(configs)

    for k in category_tag_set_dict.keys():
        category_tag_set_dict[k] = list(category_tag_set_dict[k])
    other_tag_set = list(other_tag_set)

    all_category_tags = category_tag_set_dict["상의"] + category_tag_set_dict["하의"] + \
            category_tag_set_dict["원피스"] + category_tag_set_dict["아우터"] \

    all_tags = all_category_tags + other_tag_set
            

    COUNT_FOUND = 0
    COUNT_TOTAL = 0
    COUNT_BLOCKED_STRONG = 0
    COUNT_BLOCKED_WEAK = 0
    for i, r in enumerate(dataset):
        COUNT_TOTAL += 1

        while True:
            try:
                tag_list = []
                
                response, tag_indice = sentence_to_tag(r[1], all_tags) #+ other_tag_set)
                # make unique tag list
                CATEGORY_TAG_LEN = len(all_category_tags)
                category_tag_indice = [ti for ti in tag_indice if ti < CATEGORY_TAG_LEN]
                style_tag_indice = [ti - CATEGORY_TAG_LEN for ti in tag_indice if ti >= CATEGORY_TAG_LEN]

                tag_list = tag_list + list(set([all_category_tags[ti] for ti in category_tag_indice]))\
                            + list(set([other_tag_set[ti] for ti in style_tag_indice]))

                gt_tag_list = [gt.strip() for gt in r[2].split(",")]
                tag_wo_colon_list = ["".join(t.split(":")) for t in tag_list]

                # It is not necessary to check "t in vocab.keys()", because all tags is from 
                # all_category_tags and other_tag_set
                # for t in tag_wo_colon_list:
                #     assert(t in vocab.keys())

                search_tag_query = " ".join(tag_wo_colon_list)

                retrieval_result = retriever.invoke(search_tag_query)

                img_id = r[5].split("/")[-1].split(".")[0].strip()

                time.sleep(1)


                found_index = -1
                for j, found_item in enumerate(retrieval_result):
                    found_image_id = found_item.metadata["image_path"].split("/")[-1].split(".")[0].strip()
                    # print(found_image_id, img_id)
                    if found_image_id == img_id:
                        COUNT_FOUND += 1
                        found_index = j+1
                        break

                
                BLOCKED_STRONG = False
                BLOCKED_WEAK = False
                if found_index == -1:
                    BLOCKED_STRONG = True
                    BLOCKED_WEAK = True
                    
                    # Regardless of correctness in tag extraction, if top 1 result contains ground truth tags
                    # It is BLOCKED_WEAK
                    # For compatibility with dense retriever.
                    for t in gt_tag_list:
                        if ("".join(t.split(":")) in retrieval_result[0].page_content) == False:
                            BLOCKED_WEAK = False
                            break

                    # if tag extraction is correct "and" top 1 result contains ground truth tags
                    # It is BLOCKED_STRONG

                    for t in tag_wo_colon_list:
                        if (t in retrieval_result[0].page_content) == False:
                            BLOCKED_STRONG = False
                            break
                    
                    if len(tag_list) != len(gt_tag_list):
                        BLOCKED_STRONG = False
                    else:
                        for t in tag_list:
                            if (t in gt_tag_list) == False:
                                BLOCKED_STRONG = False
                                break
                        
                    if BLOCKED_WEAK:
                        COUNT_BLOCKED_WEAK += 1

                    if BLOCKED_STRONG:
                        COUNT_BLOCKED_STRONG += 1

                    
                print(f"Found top {found_index}", COUNT_FOUND, "/", COUNT_TOTAL, COUNT_BLOCKED_STRONG, COUNT_BLOCKED_WEAK)
                print(i, found_index, category, tag_list, gt_tag_list, BLOCKED_STRONG, BLOCKED_WEAK)
                
            except ValueError as e:
                print("An error occured", e)
                continue
            else:
                break


retriever_cache_find_image = None
vocab_cache_find_image = None

def find_image(txt, configs, return_found_tags=False, return_image_as_url=False,\
               return_bucket_url=False,
               use_retriever_cache=True, num_images=30):
    global retriever_cache_find_image, vocab_cache_find_image

    t = time.time()

    retriever, vocab = (None, None)

    if use_retriever_cache and \
        (retriever_cache_find_image is None or vocab_cache_find_image is None):
        retriever_cache_find_image, vocab_cache_find_image = utils.get_full_tag_retriever(configs)
    elif use_retriever_cache == False:
        retriever_cache_find_image, vocab_cache_find_image = utils.get_full_tag_retriever(configs)

    retriever, vocab = retriever_cache_find_image, vocab_cache_find_image

    print("1", time.time() - t)
    t = time.time()

    output = sentence_to_category(txt)
    print("2", time.time() - t)
    t = time.time()
    output_indice = list(map(int, output.strip().split(",")))
    # print(gt_category_list, output_indice, output)

    category_tag_list = ["상의", "하의", "원피스", "아우터"]
    
    category_tag_set_dict, other_tag_set = load_category_to_tag_set_dict(configs)
    print("3", time.time() - t)
    t = time.time()

    for k in category_tag_set_dict.keys():
        category_tag_set_dict[k] = sorted(list(category_tag_set_dict[k]))
    other_tag_set = sorted(list(other_tag_set))


    tag_list = []
    for output_index in output_indice:
        assert(output_index > 0 and output_index < 5)

        category = category_tag_list[output_index - 1]
        last_subtag_list = ["-".join(t.split(":")[1:]) for t in category_tag_set_dict[category] + other_tag_set]
        response, tag_indice = sentence_to_tag(txt, last_subtag_list)
        # make unique tag list
        CATEGORY_TAG_LEN = len(category_tag_set_dict[category])
        category_tag_indice = [ti for ti in tag_indice if ti < CATEGORY_TAG_LEN]
        style_tag_indice = [ti - CATEGORY_TAG_LEN for ti in tag_indice if ti >= CATEGORY_TAG_LEN]

        tag_list = tag_list + list(set([category_tag_set_dict[category][ti] for ti in category_tag_indice]))\
                    + list(set([other_tag_set[ti] for ti in style_tag_indice]))
    print("4", time.time() - t)
    tt = time.time()
    tag_wo_colon_list = ["".join(t.split(":")) for t in tag_list]

    # for t in tag_wo_colon_list:
    #     assert(t in vocab.keys())
    search_tag_query = " ".join(tag_wo_colon_list)
    print(search_tag_query)
    old_top_k = retriever.k
    retriever.k = num_images


    # result = retriever.invoke(search_tag_query)
    # result = retriever._get_relevant_documents(search_tag_query, run_manager=None)

    result = retrieve_using_kernel(retriever, search_tag_query, configs)

    print("5", time.time() - tt)
    t = time.time()
    retriever.k = old_top_k

    found_image_tags_list = []
    
    print(result)
    image_list = []
    for i in range(num_images):
        image_list.append(utils.get_url_output(result[i].metadata['image_path'], return_image_as_url, return_bucket_url, configs))
        found_image_tags_list.append(result[i].metadata['image_tag_set'])

    if return_found_tags:
        return image_list, search_tag_query, found_image_tags_list
    else:
        return image_list, search_tag_query
    
if __name__ == "__main__":
    # test_sentence_to_category()
    # category_tag_set_dict, other_tag_set = load_category_to_tag_set_dict()
    # test_sentence_to_tag()

    # test_decide_style_necessity()

    # test_flat_retrieval()
    configs = json.load(open("configs.json", "r"))
    # test_structural_retrieval(configs)
    import pdb
    pdb.set_trace()
    
