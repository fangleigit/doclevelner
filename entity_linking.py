import itertools
import os
import pickle

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from tqdm import tqdm

from datasetreader.conll2003_doc import _is_doc_divider, _is_line_divider

key = ""
endpoint = ""


# Authenticate the client using your key and endpoint
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=ta_credential)
    return text_analytics_client


client = authenticate_client()
whitelist = set(line.strip().lower() for line in open('PerLocOrg.txt'))


def entity_linking():
    results = {}
    for file_path in os.listdir('CoNLL2003'):
        with open(os.path.join('CoNLL2003', file_path), "r") as data_file:
            for _is_divider, lines in tqdm(itertools.groupby(data_file, _is_doc_divider)):
                if not _is_divider:
                    fields = [line.strip().split()
                              for line in lines if not _is_line_divider(line)]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, pos_tags, chunk_tags, ner_tags = fields
                    res = entity_linking_on_tokens(tokens_)
                    if res:
                        results[' '.join(tokens_)] = res
    with open('CoNLL2003/entitylinked.pkl', "wb") as ofile:
        pickle.dump(results, ofile)


def build_doc_from_tokens(tokens):
    offset2id = {}
    content = ""
    for i, token in enumerate(tokens):
        offset2id[len(content)] = i
        if i == len(tokens)-1:
            content += token
        else:
            content += (token + " ")
    return content, offset2id


def entity_linking_on_tokens(tokens):
    try:
        linking_result = [0] * len(tokens)
        content, offset2id = build_doc_from_tokens(tokens)
        result = client.recognize_linked_entities(documents=[content])[0]
        for entity in result.entities:
            if entity.name.lower() not in whitelist:
                continue
            # print("Name: ", entity.name)
            for match in entity.matches:
                # print("Text:", match.text)
                if match.offset not in offset2id:
                    continue
                start_token_id = offset2id[match.offset]
                end_token_id = start_token_id

                cur_len = len(tokens[end_token_id])
                while cur_len < match.length:
                    end_token_id += 1
                    cur_len += len(tokens[end_token_id])+1
                # token level match
                if cur_len == match.length:
                    for idx in range(start_token_id, end_token_id+1):
                        linking_result[idx] = 1
                # print(tokens[start_token_id:end_token_id+1])
        return linking_result
    except Exception as err:
        print("Encountered exception. {}".format(err))
        return None


entity_linking()
