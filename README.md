# image-retriever-keyword-based

## Prepare data
```bash
cd app
bash scripts/0_prepare_dataset.sh
```
or
- Download `retriever_full_tag_norm_None.pkl.zip` from keitiar.com:7200, `shared_data > mmc2_data > image-retriever-tag-based/`.
- unzip the file to `./tmp/`

## Run this application using docker compose

1. set environment variables in `.env` files. You can find a sample file(`sample.env`) in the root directory.
```bash
GROQ_API_KEY="abcd"
```

2. run the application using `docker compose`
```bash
docker compose up
```

## Stop this application (stop all container)

`Ctrl` + `C`


## Down this application (remove all container)
```bash
docker compose down
```