from overrides import overrides
from typing import Dict, List, Set
import numpy as np

from allennlp.common.util import JsonDict,sanitize 
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField,IndexField

@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage" : passage, "question" : question})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)

    @overrides
    def predictions_to_labels(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> List[Instance]:
        """
        TODO
        """
        #span_start = np.argmax(outputs['span_start_logits'])
        #span_end = np.argmax(outputs['span_end_logits'])
        #print(instance)
        print(outputs)
        span_ind = outputs['best_span']
        seq = instance["passage"].tokens
        instance.add_field('span_start', IndexField(int(span_ind[0]),seq))
        instance.add_field('span_end', IndexField(int(span_ind[1]),seq))
        #instance.add_field('label_logits', MetadataField(outputs['label_logits']))
        return [instance]

    @overrides
    def attack_from_json(self, inputs: JsonDict, target_field: str = "question", ignore_tokens:List[str] = ["@@NULL@@"]) -> Dict[str, np.ndarray]:
        """
        TODO
        """
        #return self.LOO(inputs)
        new_instances = self.get_model_predictions(inputs)
        span_start,span_end = new_instances[0]["span_start"].sequence_index, new_instances[0]["span_end"].sequence_index
        print("span_start: %d, span_end: %d"%(span_start, span_end))
        new_start = span_start
        new_end = span_end
        logits = 0

        #grads,outputs = self.get_gradients(new_instances)
        #print(grads["grad_input1"].shape, grads["grad_input2"].shape)

        # handling ignore tokens
        ignore_tokens = set(ignore_tokens)
        num_ignore_tokens = 0
        for token in new_instances[0][target_field].tokens:
            if token in ignore_tokens:
                num_ignore_tokens += 1
        last_tokens = new_instances[0][target_field].tokens

        while (len(new_instances[0][target_field])>=num_ignore_tokens) :
          last_start = new_start
          last_end = new_end
          #last_logits = logits
          #new_instances[0].fields.pop('label_logits', None)

          grads,outputs = self.get_gradients(new_instances)
          model_output = self._model.decode(outputs)
          #logits = model_output["label_logits"].detach().cpu().numpy()[0]
          #new_label = np.argmax(logits)
          #print(model_output["best_span"][0].detach().cpu().numpy())
          span_ind = model_output["best_span"][0].detach().cpu().numpy()
          new_start,new_end = span_ind[0], span_ind[1]
          print("original question:", inputs[target_field])
          #print(model_output)
          print("current quesiton:",new_instances[0][target_field].tokens,"new span start:", new_start, "new span end:",new_end)
          print("------------------------")
          if (new_start!=span_start or new_end!=span_end):
            #print(last_tokens)
            break
          #print(grads.keys())
          last_tokens = list(new_instances[0][target_field].tokens)
          new_instances = self.pathological_attack(grads["grad_input2"], new_instances, target_field, ignore_tokens)
          #print(new_instances[0])

        print("final adv:", last_tokens, "new span start:", new_start, "new span end:",new_end)
        # TODO: return something else
        return sanitize({"final": last_tokens})

    def pathological_attack(self, grads:np.ndarray, instances:List[Instance], target_field: str = "hypothesis", ignore_tokens:Set[str] = {"@@NULL@@"}) -> List[Instance]:     
        """
        TODO
        """
        num_of_words = grads.shape[0]
        grads_mag = []
        for i in range(num_of_words):
            norm = np.sqrt(grads[i].dot(grads[i]))
            grads_mag.append(norm)
        #print(grads_mag)

        smallest = np.argmin(grads_mag)
        #print(ignore_tokens,str(instances[0][target_field].tokens[smallest]),type(instances[0][target_field].tokens[smallest]),instances[0][target_field].tokens[smallest] in ignore_tokens)
        while str(instances[0][target_field].tokens[smallest]) in ignore_tokens:
            print(smallest,float("inf"))
            grads_mag[smallest] = float("inf")
            smallest = np.argmin(grads_mag)

        #print("deleted word", instances[0][target_field].tokens[smallest])

        sentence_tensor = instances[0][target_field].tokens
        del sentence_tensor[smallest]
        instances[0][target_field].tokens = sentence_tensor
        instances[0].indexed = False
        return instances  
