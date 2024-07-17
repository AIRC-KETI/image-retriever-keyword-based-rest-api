import os
import requests
from io import BytesIO
from PIL import Image

import csv
import json
from langchain_community.retrievers import TFIDFRetriever

from langchain_core.documents import Document
import json

def tags_given_id(img_id:int):
    tag_json_file = open(f"/mnt/nfs4/byunggill/multimodal/k_fashion/fashion_retriever/data/k_fashion_labels/{img_id}.json", "r")
    tag_json = json.load(tag_json_file)

    img_labelings = tag_json['데이터셋 정보']['데이터셋 상세설명']['라벨링']

    colon_concated_tags = []

    for k, v in img_labelings.items():
        if isinstance(v, list):
            for d in v:
                for kk, vv in d.items():
                    if isinstance(vv, list):
                        for vvv in vv:
                            colon_concated_tags.append(":".join([k, kk, vvv]))
                    else:
                        colon_concated_tags.append(":".join([k, kk, vv]))
        else:
            continue

    tag_json_file.close()

    return colon_concated_tags


def get_full_tag_retriever(configs):
    LOAD_PKL = configs["use_retriever_cache"]
    label_dir_path = configs["label_dir_path"]
    index_file_path = configs["index_file_path"]
    retriever_cache_path = configs["retriever_cache_path"]
    tfidf_params = configs["tfidf_params"]

    TAG_CONCATENATOR=""
    vocab={}
    if LOAD_PKL == False:
        document_list = []
        kfashion_images_group_reader = csv.reader(open(index_file_path, "r"), delimiter="\t")
        for i, row in enumerate(kfashion_images_group_reader):

            img_id = row[1].split("/")[-1].split(".")[0]
            tag_json = json.load(open(f"{label_dir_path}/{img_id}.json", "r"))

            img_labelings = tag_json['데이터셋 정보']['데이터셋 상세설명']['라벨링']

            concated_tags = []
            colon_concated_tags = []

            for k, v in img_labelings.items():
                if isinstance(v, list):
                    for d in v:
                        for kk, vv in d.items():
                            if isinstance(vv, list):
                                for vvv in vv:
                                    concated_tags.append(TAG_CONCATENATOR.join([k, kk, vvv]))
                                    colon_concated_tags.append(":".join([k, kk, vvv]))

                                    if (concated_tags[-1] in vocab.keys()) == False:
                                        vocab[concated_tags[-1]] = len(vocab.keys())
                            else:
                                concated_tags.append(TAG_CONCATENATOR.join([k, kk, vv]))
                                colon_concated_tags.append(":".join([k, kk, vv]))
                                if (concated_tags[-1] in vocab.keys()) == False:
                                    vocab[concated_tags[-1]] = len(vocab.keys())

            
            page_content = " ".join(concated_tags)
            
            document_list.append(Document(page_content=page_content, metadata={'image_path': row[1], 'query':"EMPTY for compatibility", 'image_tag_set':",".join(colon_concated_tags)}))
            
            if i % 100 == 0:
                print(i)

        tfidf_params["vocabulary"] = vocab
        retriever = TFIDFRetriever.from_documents(document_list, tfidf_params=tfidf_params)
        retriever.save_local(retriever_cache_path)
    else:
        print(f"Load retriever cache: {retriever_cache_path}")
        retriever = TFIDFRetriever.load_local(f"{retriever_cache_path}", allow_dangerous_deserialization=True)
        vocab = retriever.vectorizer.vocabulary
    
    assert retriever.vectorizer.use_idf == tfidf_params["use_idf"], f"retriever use_idf does not match, {retriever.vectorizer.use_idf}, {tfidf_params['use_idf']}"
    assert retriever.vectorizer.binary == tfidf_params["binary"], f"retriever binary does not match, {retriever.vectorizer.binary}, {tfidf_params['binary']}"
    assert retriever.vectorizer.analyzer == tfidf_params["analyzer"], f"retriever analyzer does not match, {retriever.vectorizer.analyzer}, {tfidf_params['analyzer']}"
    assert retriever.vectorizer.norm == tfidf_params["norm"], f"retriever norm does not match, {retriever.vectorizer.norm}, {tfidf_params['norm']}"

    return retriever, vocab


def get_url_output(url, return_image_as_url, return_bucket_url, config):
    img_id = url.split("/")[-1].split(".")[0]
    local_image_dir_abs_path = config["local_image_dir_abs_path"]
    bucket_image_dir_path = config["bucket_image_dir_path"]
    local_url = f"{local_image_dir_abs_path}/{img_id}.jpg"
    bucket_url = f"{bucket_image_dir_path}/{img_id}.jpg"
    
    if return_image_as_url:
        return bucket_url if return_bucket_url else local_url
    else:
        if os.path.exists(local_url):
            return Image.open(local_url)
        else:
            return Image.open(BytesIO(requests.get(bucket_url).content))

if __name__ == "__main__":
    import json
    configs = json.load(open("configs.json", "r"))
    configs["use_retriever_cache"] = False
    get_full_tag_retriever(configs)
