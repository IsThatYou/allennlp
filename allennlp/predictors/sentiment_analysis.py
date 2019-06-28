from overrides import overrides
from typing import Dict, List
import numpy
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer

@Predictor.register('sentiment-analysis')
class SentimentAnalysisPredictor(Predictor):
    def predict(self, tokens: str) -> JsonDict:
        return self.predict_json({"tokens": tokens})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict["tokens"]        
        # the dataset reader for SST expects tokenized inputs
        tokenizer = WordTokenizer() 
        tokens = [str(t) for t in tokenizer.tokenize(tokens)]        
        return self._dataset_reader.text_to_instance(tokens)

    # converts the model's predictions to AllenNLP instance's with the label predicted from the model. For classification,
    # this is simply the argmax of the model's output probabilities. 
    @overrides        
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, numpy.ndarray]) -> List[Instance]:        
        label = numpy.argmax(outputs['probs'])
        instance.add_field('label', LabelField(int(label), skip_indexing=True))                    
        return [instance]
