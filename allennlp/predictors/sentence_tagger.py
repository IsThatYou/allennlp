from overrides import overrides
import copy
from typing import List, Dict 
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import ListField, LabelField


@Predictor.register('sentence-tagger')
class SentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance(tokens)
    
    @overrides
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        tags = outputs['tags']
        tag_list = []

        i = 0
        while (i < len(tags)): 
            tag = tags[i]
            if tag[0] == 'U':
                cur_tags = [t if idx == i else 'O' for idx, t in enumerate(tags)]
                tag_list.append(cur_tags)
            elif tag[0] == 'B':
                begin_idx = i
                # append a sequence vector 
                while (tag[0] != 'L'):
                    i += 1
                    tag = tags[i]
                end_idx = i
                cur_tags = [t if idx >= begin_idx and idx <= end_idx else 'O' for idx, t in enumerate(tags)]
                tag_list.append(cur_tags)
            i += 1

        instance_list = []
        for tags in tag_list:
            new_instance = copy.deepcopy(instance)
            new_instance.add_field('tags', ListField([LabelField(tag) for tag in tags]), self._model.vocab)
            instance_list.append(new_instance)
            print('NEW INSTANCE')
            print('------------')
            print(new_instance)

        return instance_list 
        

