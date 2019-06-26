
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('sentiment-analysis')
class BiattentiveClassificationPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.BiattentiveClassificationNetwork` model.
    """

    def predict(self, sentence: str) -> JsonDict:
        """
        Predicts the sentiment of the input text.

        Parameters
        ----------
        sentence : ``str``
            The input text.
        
        Returns
        -------
        A dictionary where the key "class_probabilities" determines the probabilities of each
        of the sentiment labels [0,1,2,3,4].
        """
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """        
        input_text = json_dict["sentence"]        
        return self._dataset_reader.text_to_instance(input_text)