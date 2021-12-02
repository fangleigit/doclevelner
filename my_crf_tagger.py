from typing import Any, Dict, List

import allennlp.nn.util as util
import torch
from allennlp.models.crf_tagger import CrfTagger
from overrides import overrides


class MyCrfTagger(CrfTagger):
    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        
        ##############################The only change here########################################        
        embedded_text_input = self.text_field_embedder({**tokens, **kwargs})
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, tags, mask)

            # It's not clear why, but pylint seems to think `log_likelihood` is tuple
            # (in fact, it's a torch.Tensor), so we need a disable.
            output["loss"] = - \
                log_likelihood  # pylint: disable=invalid-unary-operand-type

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask.float())
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output
