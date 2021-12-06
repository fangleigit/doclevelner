from typing import Dict, List, Iterable
import logging
import json

from overrides import overrides
from tqdm import tqdm
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def is_valid(ner_tag, start, end):
    for idx in range(start, end):
        if ner_tag[idx] != 'O':
            return False
    return True


def build_BIO_tags(record):
    ner_tags = [['O'] * len(sent) for sent in record['sents']]
    for entity in record['vertexSet']:
        for mention in entity:
            entity_type = mention['type']
            if entity_type not in ['ORG', 'LOC', 'PER']:
                continue
            sent_id = mention['sent_id']
            start, end = mention['pos']
            if is_valid(ner_tags[sent_id], start, end):
                for idx in range(start, end):
                    ner_tags[sent_id][idx] = 'B-' + \
                        entity_type if idx == start else 'I-' + entity_type
    return ner_tags


@DatasetReader.register("docred")
class DocredDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 coding_scheme: str = 'BIO',
                 lazy: bool = False,
                 label_namespace: str = "labels"
                 ):
        super().__init__(lazy)
        self._original_coding_scheme = 'BIO'
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache

        import os.path as path
        import pickle
        with open(path.join(path.dirname(file_path), 'entitylinked.pkl'), 'rb') as ifile:
            entity_linked = pickle.load(ifile)

        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)
            records = json.load(data_file)

            for record in tqdm(records):
                ner_tags_ = build_BIO_tags(record)
                doc_tokens_ = [item for sent in record['sents']
                               for item in sent]
                entity_linked_tokens = [0] * len(doc_tokens_)
                if ' '.join(doc_tokens_) in entity_linked:
                    entity_linked_tokens = entity_linked[' '.join(doc_tokens_)]

                if not self.is_sent_input:
                    tokens = [Token(token) for token in doc_tokens_]
                    ner_tags = [tag for tags_ in ner_tags_ for tag in tags_]
                    yield self.text_to_instance(tokens, ner_tags, entity_linked_tokens)
                else:
                    doc_token_start_idx = 0
                    for sent, ner_tag_ in zip(record['sents'], ner_tags_):
                        tokens = [Token(token) for token in sent]
                        doc_token_end_idx = doc_token_start_idx + len(tokens)
                        sent_entity_linked_tokens = entity_linked_tokens[
                            doc_token_start_idx:doc_token_end_idx]
                        yield self.text_to_instance(tokens, ner_tag_, sent_entity_linked_tokens)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         ner_tags: List[str] = None,
                         entity_linked_tokens: List[int] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields["metadata"] = MetadataField(
            {"words": [x.text for x in tokens]})

        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL":
            coded_ner = to_bioul(ner_tags,
                                 encoding=self._original_coding_scheme) if ner_tags is not None else None
        else:
            coded_ner = ner_tags

        # Add entitylinked feature
        entity_linked_tokens = [str(item) for item in entity_linked_tokens]
        instance_fields['el_tags'] = SequenceLabelField(
            entity_linked_tokens, sequence, "el_tags")

        # Add "tag label" to instance
        instance_fields['tags'] = SequenceLabelField(coded_ner, sequence,
                                                     self.label_namespace)

        return Instance(instance_fields)
