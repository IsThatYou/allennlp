from typing import Dict, List, Set
import numpy
from allennlp.attack import Attacker
from allennlp.common.util import JsonDict, sanitize 
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields.field import DataArray, Field

@Attacker.register('pathologies')
class Pathologies(Attacker):
    def __init__(self, predictor):
        super().__init__(predictor)
    def attack_from_json(self, inputs:JsonDict, target_field: str = "hypothesis", ignore_tokens:List[str] = ["@@NULL@@"]):
        '''
        TODO
        '''
        new_instances = self.predictor.inputs_to_labeled_instances(inputs)
        label = new_instances[0]["label"].label
        print("start label:",label)
        new_label = label
        logits = 0

        # handling ignore tokens
        ignore_tokens = set(ignore_tokens)
        num_ignore_tokens = 0
        for token in new_instances[0][target_field].tokens:
            if token in ignore_tokens:
                num_ignore_tokens += 1
        last_tokens = new_instances[0][target_field].tokens

        while (len(new_instances[0][target_field])>=num_ignore_tokens) :
          last_label = new_label
          last_logits = logits
          #new_instances[0].fields.pop('label_logits', None)

          grads,outputs = self.predictor.get_gradients(new_instances)
          model_output = self.predictor._model.decode(outputs)
          logits = model_output["label_logits"].detach().cpu().numpy()[0]
          new_label = numpy.argmax(logits)
          print("label:", new_label, "logits:",logits)
          print("------------------------")
          if (new_label!=label):
            #print(last_tokens)
            break
          print(grads.keys())
          last_tokens = list(new_instances[0][target_field].tokens)
          new_instances = self.pathological_attack(grads["grad_input_1"], new_instances, target_field, ignore_tokens)
          print(new_instances[0])

        print("final adv:", last_tokens," | label:",last_label," | logits:", last_logits)
        # TODO: return something else
        return sanitize({"final": last_tokens})

    def pathological_attack(self, grads:numpy.ndarray, instances:List[Instance], target_field: str = "hypothesis", ignore_tokens:Set[str] = {"@@NULL@@"}) -> List[Instance]:     
        """
        TODO
        """
        num_of_words = grads.shape[0]
        grads_mag = []
        for i in range(num_of_words):
            norm = numpy.sqrt(grads[i].dot(grads[i]))
            grads_mag.append(norm)
        #print(grads_mag)

        smallest = numpy.argmin(grads_mag)
        #print(ignore_tokens,str(instances[0][target_field].tokens[smallest]),type(instances[0][target_field].tokens[smallest]),instances[0][target_field].tokens[smallest] in ignore_tokens)
        while str(instances[0][target_field].tokens[smallest]) in ignore_tokens:
            print(smallest,float("inf"))
            grads_mag[smallest] = float("inf")
            smallest = numpy.argmin(grads_mag)

        #print("deleted word", instances[0][target_field].tokens[smallest])

        sentence_tensor = instances[0][target_field].tokens
        del sentence_tensor[smallest]
        instances[0][target_field].tokens = sentence_tensor
        instances[0].indexed = False
        return instances  