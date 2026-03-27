import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.stats import bootstrap, kendalltau
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.proxann.llm_annotations.utils import process_responses

data_jsons = [
    "./data/json_out/config_wiki_part1.json",
    "./data/json_out/config_wiki_part2.json",
    "./data/json_out/config_bills_part1.json",
    "./data/json_out/config_bills_part2.json",
]
response_csvs = [
    "./data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank+-+Bills_December+14,+2024_13.20.csv",
    "./data/human_annotations/Cluster+Evaluation+-+Sort+and+Rank_December+12,+2024_05.19.csv",
]
qwen_paths = {
    "wiki": "./data/llm_out/mean/wiki/Qwen3-8B/q1_then_q3_mean,q1_then_q2_mean_temp1.0_0.0_0.0_seed34_20250529_1804/llm_results_q1.json",
    "bills": "./data/llm_out/mean/bills/Qwen3-8B/q1_then_q3_mean,q1_then_q2_mean_temp1.0_0.0_0.0_seed34_20250529_1808/llm_results_q1.json",
}
start_date = "2024-12-06 09:00:00"


encoder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-nano",
    trust_remote_code=True,
    device="cpu",
    model_kwargs={"dtype": torch.bfloat16},  # Recommended for GPUs
    # config_kwargs={
    #     "_attn_implementation": "flash_attention_2"
    # },  # Recommended but optional
)

encoder = SentenceTransformer("all-MiniLM-L6-v2")


qwen_names = {}
for dataset, path in qwen_paths.items():
    with Path(path).open() as in_file:
        data = json.loads(in_file.read())
        for topic_data in data:
            topic_name = topic_data["categories"][0].removeprefix("CATEGORY: ")
            qwen_names[(dataset, topic_data["model"], topic_data["topic"])] = topic_name


def short_name(model_name):
    if "ctm" in model_name:
        return "ctm"
    if "bertopic" in model_name:
        return "bertopic"
    if "mallet" in model_name:
        return "mallet"
    else:
        return model_name


def get_topic_embeddings(topic_data):
    keywords = topic_data["topic_words"]
    docs = [doc["text"] for doc in topic_data["exemplar_docs"]]
    return encoder.encode(
        keywords + docs, task="retrieval", prompt_name="query", show_progress_bar=True
    )


def maxsim(topic_embeddings, document_embeddings):
    return np.max(cosine_similarity(topic_embeddings, document_embeddings), axis=0)


def rank(a):
    return len(a) - np.argsort(np.argsort(a))


responses = {}
for csv in response_csvs:
    for topic_id, topic_responses in process_responses(
        csv,
        data_jsons,
        start_date=start_date,
        path_save=None,
        removal_condition="loose",
    ).items():
        if topic_responses:
            responses[topic_id] = topic_responses


human_records = []
for data_json in data_jsons:
    data_json = Path(data_json)
    print(f"Processing file {data_json}")
    with data_json.open() as in_file:
        data = json.loads(in_file.read())
    _, dataset, _ = data_json.stem.split("_")
    _dataset = dataset
    if dataset == "wiki":
        _dataset = "wikitext"
    for model_id, model_data in data.items():
        model_name = short_name(model_id)
        print(f"----{model_name}----")
        total = len(model_data)
        for topic_id, topic_data in tqdm(
            model_data.items(), desc="Going through topics", total=total
        ):
            eval_docs = topic_data["eval_docs"]
            doc_ids = [doc["doc_id"] for doc in eval_docs]
            text = [doc["text"] for doc in eval_docs]
            document_embeddings = encoder.encode(
                text, show_progress_bar=True, task="retrieval", prompt_name="document"
            )
            topic_embeddings = get_topic_embeddings(topic_data)
            qwen_name = qwen_names[(dataset, model_name, topic_id)]
            name_embedding = encoder.encode(
                [qwen_name], task="retrieval", prompt_name="query"
            )
            named_rank = rank(cosine_similarity(name_embedding, document_embeddings)[0])
            named_rank = dict(zip(doc_ids, named_rank))
            maxsim_rank = rank(maxsim(topic_embeddings, document_embeddings))
            maxsim_rank = dict(zip(doc_ids, maxsim_rank))
            mean_rank = rank(
                cosine_similarity(
                    [np.mean(topic_embeddings, axis=0)], document_embeddings
                )[0]
            )
            mean_rank = dict(zip(doc_ids, mean_rank))
            human_data = responses[f"{_dataset}-labeled/{model_name}/{topic_id}"]
            for i_human, human in enumerate(human_data):
                for entry in human["eval_docs"]:
                    human_records.append(
                        dict(
                            dataset=dataset,
                            topic_id=topic_id,
                            model_id=short_name(model_id),
                            doc_id=entry["doc_id"],
                            annotator_id=f"human_{i_human}",
                            document_probability=entry["prob"],
                            rank=entry["rank"],
                            maxsim_rank=maxsim_rank[entry["doc_id"]],
                            mean_rank=mean_rank[entry["doc_id"]],
                            named_rank=named_rank[entry["doc_id"]],
                        )
                    )
encoder_df = pd.DataFrame.from_records(human_records)

encoder_df.to_csv("encoder_results.csv")

summary = []
for (dataset, topic_id), subdata in encoder_df.groupby(["dataset", "topic_id"]):
    for rank_type in ["mean", "maxsim", "named"]:
        tau, p = kendalltau(subdata["rank"], subdata[f"{rank_type}_rank"])
        summary.append(
            dict(dataset=dataset, topic_id=topic_id, tau=tau, p=p, rank_type=rank_type)
        )
summary = pd.DataFrame.from_records(summary)


fig = make_subplots(
    cols=2,
    rows=1,
    subplot_titles=["wiki", "bills"],
)
human_human = [0.55, 0.42]
for (dataset, rank_type), subdata in summary.groupby(["dataset", "rank_type"]):
    col = 0 if dataset == "wiki" else 1
    tau = np.array(subdata["tau"])
    estimate = np.mean(tau)
    (low, high) = bootstrap((tau,), np.mean).confidence_interval
    error = estimate - low, high - estimate
    fig.add_bar(
        x=[rank_type],
        y=[estimate],
        error_y=dict(
            type="data", array=[error[1]], arrayminus=[error[0]], symmetric=False
        ),
        col=col + 1,
        row=1,
    )
    if rank_type == "maxsim":
        fig.add_hline(
            y=human_human[col], line=dict(color="gray", dash="dash"), col=col + 1, row=1
        )
fig = fig.update_layout(template="plotly_white")
fig.show()
