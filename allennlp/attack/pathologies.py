from typing import Dict, List, Set
import numpy
import torch
from allennlp.attack import Attacker
from allennlp.common.util import JsonDict, sanitize 
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields import IndexField

from collections import defaultdict

@Attacker.register('pathologies')
class Pathologies(Attacker):
    def __init__(self, predictor):
        super().__init__(predictor)
    def attack_from_json(self, inputs:JsonDict, target_field: str, gradient_index:str,ignore_tokens:List[str] = ["@@NULL@@"]):
        '''
        TODO
        '''
        JsonSet = set(inputs.keys())
        # print(JsonSet)
        check_fields = set()
        check_list = {}

        instance = self.predictor._json_to_instance(inputs)
        original_fields = set(instance.fields.keys())
        original_vals = {x:instance[x] for x in original_fields}
        og_instances = self.predictor.inputs_to_labeled_instances(inputs)
        final_tokens = []
        for i in range(len(og_instances)):
            new_instances = [og_instances[i]]
            new_fields = set(new_instances[0].fields.keys())
            #check_fields = new_fields.difference(original_fields)
            # print(target_field)
            original = [x for x in new_instances[0][target_field].tokens]
            grads,outputs = self.predictor.get_gradients(new_instances)
            # print(grads)
            # print('\n\n\n')
            # print(original_fields)
            # print(new_fields)
            # print("check fields =",check_fields)
            #check_list = {x:new_instances[0][x] for x in check_fields}
            #print(check_list)
            # print(original_vals)
            test_instances = self.predictor.inputs_to_labeled_instances(inputs)
            for key in new_fields:
                if (key not in JsonSet) and (key != target_field):
                    check_fields.add(key)
                    check_list[key] = test_instances[0][key]
            # print("check fields = ",check_fields)
            # print("check list = ",check_list)

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
            if "tags" in new_instances[0]:
                og_label_list = new_instances[0]["tags"].__dict__["field_list"]
                tag_dict = defaultdict(int)
                tag_tok = ''
                og_mask = []
                og_tags = []
                for label in og_label_list:
                    #print(label)
                    #print(label.label)
                    if label.label != "O":
                        tag_dict[label.label] += 1
                        tag_tok = tag_tok + label.label
                        og_mask.append(1)
                        og_tags.append(label.label)
                        num_ignore_tokens +=1
                    else:
                        og_mask.append(0)
                #print(tag_dict)
                #print(tag_tok)
                #print(og_mask)
                #print("og_tags = ", og_tags)
            else:
                num_ignore_tokens = 1 # don't go below 1 token for classification/entailment/etc.

            #print("check fields =",check_fields)
            idx = -1
            while (len(new_instances[0][target_field])>=num_ignore_tokens) :
                #print(new_instances[0])
                #outputs = self.predictor._model.forward_on_instance(instance)
                #print(outputs)


                grads,outputs = self.predictor.get_gradients(new_instances)                
                #model_output = self.predictor._model.decode(outputs)
                # print(outputs)
                for each in outputs:
                    if isinstance(outputs[each], torch.Tensor):
                        derail = outputs[each].detach().cpu().numpy().squeeze().squeeze()
                        outputs[each] = derail
                    elif isinstance(outputs[each],list):
                        derail = outputs[each][0]
                        #print(derail)
                        #print(type(derail))
                        outputs[each] = derail
                test_instances = self.predictor.predictions_to_labeled_instances(new_instances[0],outputs)
                #print(len(new_instances[0][target_field]),"------------------------")
                label_change = False
                if "tags" not in new_instances[0]:
                    for field in check_fields:
                        #print(field)
                        # print(super(IndexField,new_instances[0][field]).__eq__(check_list[field]))
                        if field in new_instances[0].fields:
                            equal = new_instances[0][field].__eq__(check_list[field])
                            #print(equal)
                            #print(new_instances[0][field],check_list[field])
                        else:
                            equal = outputs[field] == check_list[field]
                            #print(equal)
                            # print(outputs[field],check_list[field])
                        if (not equal):
                            label_change = True
                            break
                if label_change:
                    break
                if "tags" in new_instances[0]:
                    # cur_label_list = new_instances[0]["tags"].__dict__["field_list"]
                    # print(cur_label_list)
                    # cur_tag_dict = defaultdict(int)
                    # cur_tag_tok = ''
                    # for label in cur_label_list:
                    #     if label.label != "O":
                    #         cur_tag_dict[label.label] += 1
                    #         cur_tag_tok = cur_tag_tok + label.label
                    # print(cur_tag_dict)
                    # equal = (cur_tag_dict == tag_dict) and (cur_tag_tok==tag_tok)
                    if idx!=-1:
                        del og_mask[idx]
                    print("og_mask = ",og_mask)
                    cur_tags = [outputs["tags"][x] for x in range(len(outputs["tags"])) if og_mask[x]]
                    print("cur_tags = ",cur_tags)
                    print("og_tags = ", og_tags)
                    print("length = ", )
                    equal = (cur_tags == og_tags)
                    if not equal:
                        break
                last_tokens = list(new_instances[0][target_field].tokens)                 
                #print(self.pathological_attack(grads[gradient_index], new_instances, target_field, ignore_tokens))
                new_instances,idx = self.pathological_attack(grads[gradient_index], new_instances, target_field, ignore_tokens)

            
            #print("final adv:", last_tokens)
            # TODO: return something else
            #print(original)
            final_tokens.append(last_tokens)
        #print(final_tokens)
        return sanitize({"final": final_tokens,"original":original})

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

        if "tags" in instances[0]:
            label_list = instances[0]["tags"].__dict__["field_list"]
            for idx,label in enumerate(label_list):
                if label.label != "O":
                    grads_mag[idx] = float("inf")

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

        # print("inside")
        # print(instances[0]["tags"])
        if "tags" in instances[0]:
            label_list = instances[0]["tags"].__dict__["field_list"]
            del label_list[smallest]
            instances[0]["tags"].__dict__["field_list"] = label_list
            # print(instances[0]["tags"])

        instances[0].indexed = False
        return instances,smallest  