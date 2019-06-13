from typing import Dict, List, Set
import numpy
import torch
from allennlp.attack import Attacker
from allennlp.common.util import JsonDict, sanitize 
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields import IndexField

@Attacker.register('pathologies')
class Pathologies(Attacker):
    def __init__(self, predictor):
        super().__init__(predictor)
    def attack_from_json(self, inputs:JsonDict, target_field: str, gradient_index:str,ignore_tokens:List[str] = ["@@NULL@@"]):
        '''
        TODO
        '''
        JsonSet = set(inputs.keys())
        print(JsonSet)
        check_fields = set()
        check_list = {}

        instance = self.predictor._json_to_instance(inputs)
        original_fields = set(instance.fields.keys())
        original_vals = {x:instance[x] for x in original_fields}
        new_instances = self.predictor.inputs_to_labeled_instances(inputs)
        new_fields = set(new_instances[0].fields.keys())
        #check_fields = new_fields.difference(original_fields)
        original = [x for x in new_instances[0][target_field].tokens]
        print(original_fields)
        print(new_fields)
        print("check fields =",check_fields)
        #check_list = {x:new_instances[0][x] for x in check_fields}
        #print(check_list)
        print(original_vals)


        test_instances = self.predictor.inputs_to_labeled_instances(inputs)
        # for key in new_fields:
        #     comp = test_instances[0][key].__eq__(new_instances[0][key])
            
        #     print(key)
        #     print([test_instances[0][key],new_instances[0][key]])
        #     print(test_instances[0][key].__dict__)
        #     print(new_instances[0][key].__dict__)
        #     print(comp,"------------------------------")
        #     if (not comp):
        #         check_fields.add(key)
        #         check_list[key] = new_instances[0][key]
        # print("check fields = ",check_fields)
        # print("check list = ",check_list)

        # label = new_instances[0]["label"].label
        # print("start label:",label)
        # new_label = label
        # logits = 0
        for key in new_fields:
            if key not in JsonSet:
                check_fields.add(key)
                check_list[key] = test_instances[0][key]
        print("check fields = ",check_fields)
        print("check list = ",check_list)
        
        # grads,outputs = self.predictor.get_gradients(new_instances)
        # if "answer" in outputs:
        #     check_fields.add("answer")
        #     check_list["answer"] = outputs["answer"]

        # handling ignore tokens
        ignore_tokens = set(ignore_tokens)
        num_ignore_tokens = 0
        for token in new_instances[0][target_field].tokens:
            if str(token) in ignore_tokens:
                num_ignore_tokens += 1
        last_tokens = new_instances[0][target_field].tokens
        #print(num_ignore_tokens)

        print("check fields =",check_fields)
        while (len(new_instances[0][target_field])>num_ignore_tokens) :
            grads,outputs = self.predictor.get_gradients(new_instances)
            #model_output = self.predictor._model.decode(outputs)
            print(outputs)
            for each in outputs:
                if isinstance(outputs[each], torch.Tensor):
                    derail = outputs[each].detach().cpu().numpy().squeeze().squeeze()
                    outputs[each] = derail
                elif isinstance(outputs[each],list):
                    derail = outputs[each][0]
                    #print(derail)
                    #print(type(derail))
                    outputs[each] = derail
            self.predictor.predictions_to_labeled_instances(new_instances[0],outputs)
            print(new_instances[0])
            print("------------------------")
            label_change = False
            for field in check_fields:
                print(field)
                # print(super(IndexField,new_instances[0][field]).__eq__(check_list[field]))
                if field in new_instances[0].fields:
                    equal = new_instances[0][field].__eq__(check_list[field])
                    #print(equal)
                    #print(new_instances[0][field],check_list[field])
                else:
                    equal = outputs[field] == check_list[field]
                    print(equal)
                    print(outputs[field],check_list[field])
                if (not equal):
                    label_change = True
                    break
            if label_change:
                break
            last_tokens = list(new_instances[0][target_field].tokens)
            new_instances = self.pathological_attack(grads[gradient_index], new_instances, target_field, ignore_tokens)
          

        print("final adv:", last_tokens)
        # TODO: return something else
        print(original)
        return sanitize({"final": last_tokens,"original":original})

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
        num_ignore_tokens = len(ignore_tokens)
        counter = 0
        #print(ignore_tokens,str(instances[0][target_field].tokens[smallest]),type(instances[0][target_field].tokens[smallest]),instances[0][target_field].tokens[smallest] in ignore_tokens)
        while str(instances[0][target_field].tokens[smallest]) in ignore_tokens:
            if (counter == num_ignore_tokens):
                return instances
            grads_mag[smallest] = float("inf")
            smallest = numpy.argmin(grads_mag)
            counter +=1 
            

        #print("deleted word", instances[0][target_field].tokens[smallest])

        sentence_tensor = instances[0][target_field].tokens
        del sentence_tensor[smallest]
        instances[0][target_field].tokens = sentence_tensor
        instances[0].indexed = False
        return instances  