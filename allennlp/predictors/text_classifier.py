from overrides import overrides
from typing import Dict, List
import numpy
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer

@Predictor.register('sentiment-analysis')
class TextClassifierPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the :class:`~allennlp.models.basic_classifier.BasicClassifier` model
    """
    def predict(self, tokens: str) -> JsonDict:
        return self.predict_json({"tokens": tokens})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"tokens": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """                
        tokens = json_dict["tokens"]        
        tokenizer = WordTokenizer()
        tokens = [str(t) for t in tokenizer.tokenize(tokens)]        
        return self._dataset_reader.text_to_instance(tokens)

    @overrides        
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, numpy.ndarray]) -> List[Instance]:        
        label = numpy.argmax(outputs['probs'])
        instance.add_field('label', LabelField(int(label), skip_indexing=True))                    
        return [instance]