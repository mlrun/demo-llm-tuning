import os
from pathlib import Path
from datasets import load_dataset
import json
import zipfile
import tempfile
import mlrun

ARTICLE_TOKEN = "Article: "
HEADER_TOKEN = "Subject: "
CONTENT_TOKEN = "Content: "
DATA_FORMAT = """Subject: {} {}
Content: {}"""
END_OF_ARTICLE = "Latest Posts"


def convert_textfile_to_data_with_prompts(txt_file):
    # Read file:
    with open(txt_file, "r") as f:
        lines = f.readlines()

    start = 0
    end = 0
    subject_idx = []
    data = []
    # Dividing text into header - paragraph prompts:
    for i, line in enumerate(lines):
        if not start and line.startswith(ARTICLE_TOKEN):
            start = i
        elif HEADER_TOKEN + END_OF_ARTICLE in line:
            end = i
            break
        if line.startswith(HEADER_TOKEN):
            subject_idx.append(i)
    article_content = lines[start:end]
    subject_idx = [subject_i - start for subject_i in subject_idx]
    article_name = article_content[0].replace(ARTICLE_TOKEN, "")
    for i, subject in enumerate(subject_idx):
        if subject + 1 in subject_idx:
            continue
        subject_data = article_content[subject].replace(HEADER_TOKEN, "")
        if i + 1 == len(subject_idx):
            content_end = len(article_content)
        else:
            content_end = subject_idx[i + 1]
        content_limits = subject + 1, content_end
        data.append(
            DATA_FORMAT.format(
                article_name,
                subject_data,
                "".join(article_content[content_limits[0] : content_limits[1]]),
            )
        )
    return data


@mlrun.handler(outputs=["html-data:dataset"])
def prepare_dataset(source_dir: str):
    with zipfile.ZipFile(source_dir, "r") as zip_file:
        tmp_dir = tempfile.mkdtemp()
        zip_file.extractall(tmp_dir)
        
    path_list = Path(tmp_dir).glob(f"./*.txt")
    data = []
    # Converting text files into data in our prompt format:
    for path in path_list:
        data.extend(convert_textfile_to_data_with_prompts(path))
    data_dir = tempfile.mkdtemp()
    os.makedirs(data_dir, exist_ok=True)
    with open(data_dir + "/html_data.jsonl", "w", encoding="utf8") as f:
        for item in data:
            f.write(
                json.dumps({"text": item.replace(" ", "")}, ensure_ascii=False) + "\n"
            )
    return load_dataset(data_dir)["train"].to_pandas()
