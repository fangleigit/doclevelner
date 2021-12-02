from typing import Dict, List, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import Conll2003DatasetReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_line_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    return False


def _is_doc_divider(line: str) -> bool:
    return "-DOCSTART-" in line


@DatasetReader.register("conll2003-doc")
class Conll2003DocDatasetReader(Conll2003DatasetReader):
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

            # Group into alternative divider / sentence chunks.
            for _is_divider, lines in itertools.groupby(data_file, _is_doc_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not _is_divider:
                    fields = [line.strip().split()
                              for line in lines if not _is_line_divider(line)]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, pos_tags, chunk_tags, ner_tags = fields
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens_]
                    entity_linked_tokens = [0] * len(tokens_)
                    if ' '.join(tokens_) in entity_linked:
                        entity_linked_tokens = entity_linked[' '.join(tokens_)]
                    yield self.text_to_instance(tokens, pos_tags, chunk_tags, ner_tags, entity_linked_tokens)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         pos_tags: List[str] = None,
                         chunk_tags: List[str] = None,
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
            coded_chunks = to_bioul(chunk_tags,
                                    encoding=self._original_coding_scheme) if chunk_tags is not None else None
            coded_ner = to_bioul(ner_tags,
                                 encoding=self._original_coding_scheme) if ner_tags is not None else None
        else:
            # the default IOB1
            coded_chunks = chunk_tags
            coded_ner = ner_tags

        # Add "feature labels" to instance
        if 'pos' in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError("Dataset reader was specified to use pos_tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['pos_tags'] = SequenceLabelField(
                pos_tags, sequence, "pos_tags")
        if 'chunk' in self.feature_labels:
            if coded_chunks is None:
                raise ConfigurationError("Dataset reader was specified to use chunk tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['chunk_tags'] = SequenceLabelField(
                coded_chunks, sequence, "chunk_tags")
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError("Dataset reader was specified to use NER tags as "
                                         " features. Pass them to text_to_instance.")
            instance_fields['ner_tags'] = SequenceLabelField(
                coded_ner, sequence, "ner_tags")

        # Add entitylinked feature
        entity_linked_tokens = [str(item) for item in entity_linked_tokens]
        instance_fields['el_tags'] = SequenceLabelField(
            entity_linked_tokens, sequence, "el_tags")

        # Add "tag label" to instance
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence,
                                                         self.label_namespace)
        elif self.tag_label == 'pos' and pos_tags is not None:
            instance_fields['tags'] = SequenceLabelField(pos_tags, sequence,
                                                         self.label_namespace)
        elif self.tag_label == 'chunk' and coded_chunks is not None:
            instance_fields['tags'] = SequenceLabelField(coded_chunks, sequence,
                                                         self.label_namespace)

        return Instance(instance_fields)
