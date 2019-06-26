from typing import Dict, List, Set
import numpy
import torch
from allennlp.attack import Attacker
from allennlp.common.util import JsonDict, sanitize 
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields import IndexField
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.tokenizers import Token
from collections import defaultdict

@Attacker.register('hotflip')
class Hotflip(Attacker):
    def __init__(self, predictor):
        super().__init__(predictor)
        print(predictor)
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                M = module
        print(M)
        print("indexers = ", self.predictor._dataset_reader._token_indexers)
        
        vocab = self.predictor._model.vocab
        self.vocab = vocab
        all_tokens = list(vocab._token_to_index["tokens"].keys())

        a = [x for x in vocab._index_to_token["tokens"].keys()]
        b = torch.LongTensor(a)
        b = b.unsqueeze(0)
        c = {"tokens":b}
        print("b = ",b.shape)
        # d = " ".join(all_tokens)
        # ins = self.predictor._dataset_reader.text_to_instance(d,"")
        # print("instance length: ",len(ins["question"]))
        if "token_characters" in self.predictor._dataset_reader._token_indexers:
            tokenizer = self.predictor._dataset_reader._token_indexers["token_characters"]._character_tokenizer
            print(self.predictor._dataset_reader._token_indexers["token_characters"]._min_padding_length)
            print(self.predictor._dataset_reader._token_indexers["token_characters"]._character_tokenizer._byte_encoding)
            print(all_tokens[0])
            t = tokenizer.batch_tokenize(all_tokens)
            # print(t[0][0])
            tt = tokenizer.tokenize("How")
            # print(vocab._token_to_index["tokens"]["H"])
            # print(vocab._token_to_index["tokens"]["o"])
            # print(vocab._token_to_index["tokens"]["w"])
            print("_namespace = ",self.predictor._dataset_reader._token_indexers["token_characters"]._namespace)
            index = self.vocab.get_token_index("H", self.predictor._dataset_reader._token_indexers["token_characters"]._namespace)
            index2 = self.vocab.get_token_index("o", self.predictor._dataset_reader._token_indexers["token_characters"]._namespace)
            print("index = ",index,index2)

            character_tokens = []
            pad_length = max([len(x) for x in t])
            print("length of t = ",len(t))
            print("pad_length = ", pad_length)
            if getattr(t[0][0], 'text_id', None) is not None:
                # print("t is = ",t[0][0])
                for each in t:
                    tmp = [x.text_id for x in each]
                    # print(tmp)
                    tmp = tmp + [0] * (pad_length-len(tmp))
                    character_tokens.append(tmp)
                print(character_tokens[0][0])
                print(type(character_tokens[0][0]))
                character_tokens = torch.LongTensor(character_tokens)
                c["token_characters"] = character_tokens.unsqueeze(0)
            else:
                for each in t:
                    tmp = [self.vocab.get_token_index(x.text,self.predictor._dataset_reader._token_indexers["token_characters"]._namespace) for x in each]
                    # print(tmp)
                    tmp = tmp + [0] * (pad_length-len(tmp))
                    character_tokens.append(tmp)
                print(character_tokens[0][0])
                print(type(character_tokens[0][0]))
                character_tokens = torch.LongTensor(character_tokens)
                c["token_characters"] = character_tokens.unsqueeze(0)

            if "elmo" in self.predictor._dataset_reader._token_indexers:
                pad_length = pad_length + 2
                lltokens = []
                elmo_tokens = []
                for each in all_tokens:
                    ltokens = [Token(text=each)]
                    # print(ltokens)
                    tmp = self.predictor._dataset_reader._token_indexers["elmo"].tokens_to_indices(ltokens, self.vocab,"sentence")
                    tmp = tmp["sentence"]
                    # print(tmp)
                    lltokens.append(tmp[0])
                print(len(lltokens))
                lltokens = torch.LongTensor(lltokens)
                c["elmo"] = lltokens.unsqueeze(0)
                print(len(c["elmo"][0]))
                print(len(c["tokens"][0]))
                print(len(c["token_characters"][0]))
        embedding_matrix = M(c)
        embedding_matrix = embedding_matrix.squeeze()
        #print(embedding_matrix[476])
        print("embedding matrix shape =",embedding_matrix.shape)
        self.token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=embedding_matrix.shape[1], weight=embedding_matrix,trainable=False)


    def attack_from_json(self, inputs:JsonDict, target_field: str, gradient_index:str,ignore_tokens:List[str] = ["@@NULL@@"]):
        JsonSet = set(inputs.keys())
        
        og_instances = self.predictor.inputs_to_labeled_instances(inputs)
        # label = new_instances[0]["label"].label
        # embedder = self.predictor._model._text_field_embedder._token_embedders["tokens"]
        original = list(og_instances[0][target_field].tokens)
        print(original)
        final_tokens = []
        for i in range(len(og_instances)):
            new_instances = [og_instances[i]]
            in_text_field = new_instances[0][target_field]._indexed_tokens

            # padding_length = new_instances[0][target_field].get_padding_lengths()
            # temp = new_instances[0][target_field].as_tensor(padding_length)
            # print(temp)
            # temp2 = embedder(temp["tokens"])
            # print(temp2[0])
            #print(token_embedding)
    
            # print(M._token_embedders["tokens"].weight.shape)
            # tokens = new_instances[0][target_field].tokens
            # print(tokens)
            indexed_tokens = in_text_field["tokens"]
            which_token = 0
            adv_token_idx = indexed_tokens[which_token]
            print(indexed_tokens)
            print(adv_token_idx)

            # new_instances = self.predictor.inputs_to_labeled_instances(inputs)
            print(new_instances[0])
            # handling ignore tokens by creating a mask
            ignore_tokens = set(ignore_tokens)
            num_ignore_tokens = 0
            ignore_tokens_mask = [0]*len(new_instances[0][target_field].tokens)
            for idx,token in enumerate(new_instances[0][target_field].tokens):
                if str(token) in ignore_tokens:
                    num_ignore_tokens += 1
                    ignore_tokens_mask[idx] = 1
            print(ignore_tokens_mask)

            # handling fileds that need to be checked
            check_fields = set()
            check_list = {}
            new_fields = set(new_instances[0].fields.keys())
            test_instances = self.predictor.inputs_to_labeled_instances(inputs)
            for key in new_fields:
                if (key not in JsonSet) and (key != target_field):
                    check_fields.add(key)
                    check_list[key] = test_instances[0][key]
            print("check fields = ",check_fields)
            print("check list = ",check_list)

            # checking for tags
            if "tags" in new_instances[0]:
                og_label_list = new_instances[0]["tags"].__dict__["field_list"]
                tag_dict = defaultdict(int)
                tag_tok = ''
                og_mask = []
                for label in og_label_list:
                    print(label)
                    print(label.label)
                    if label.label != "O":
                        tag_dict[label.label] += 1
                        tag_tok = tag_tok + label.label
                        og_mask.append(1)
                    else:
                        og_mask.append(0)
                print(tag_dict)
                print(tag_tok)

            last_tokens = new_instances[0][target_field].tokens

            grads,outputs = self.predictor.get_gradients(new_instances)
            print(outputs)
            # model_output = self.predictor._model.decode(outputs)
            #print("start label:",label,"logits: ",model_output["label_logits"])
            n = len(indexed_tokens)

            flipped = []
            while True:
                grad = grads[gradient_index]
                num_of_words = grad.shape[0]
                grads_mag = []
                for i in range(num_of_words):
                    norm = numpy.sqrt(grad[i].dot(grad[i]))
                    grads_mag.append(norm)
                if "tags" in new_instances[0]:
                    label_list = new_instances[0]["tags"].__dict__["field_list"]
                    for idx,label in enumerate(label_list):
                        if label.label != "O":
                            grads_mag[idx] = -1
                    for idx in flipped:
                        grads_mag[idx] = -1
                print(num_of_words, grads_mag)
                which_token = numpy.argmax(grads_mag)
                if grads_mag[which_token] == -1:
                    break
                flipped.append(which_token)
                adv_token_idx = new_instances[0][target_field]._indexed_tokens["tokens"][which_token]
                print("which_token = ",which_token,"adv_token_idx = ",adv_token_idx)

                # last_tokens = list(new_instances[0][target_field].tokens) 
                ret = self.hotflip_attack(grad[which_token], self.token_embedding.weight, adv_token_idx)
                ret = ret.data[0].detach().cpu().item()
                print("flipped token id: ",ret)
                i2t = self.vocab._index_to_token["tokens"]
                print("flipped token: ",i2t[ret])
                print("previous sentence: ",new_instances[0][target_field].tokens)
                new_instances[0][target_field].tokens[which_token] = Token(i2t[ret])
                new_instances[0].indexed = False
                print("afterwards: ",new_instances[0][target_field].tokens)
                print("previous index: ",new_instances[0][target_field]._indexed_tokens)

                grads,outputs = self.predictor.get_gradients(new_instances)
                # model_output = self.predictor._model.decode(outputs)
                # print(model_output)
                for each in outputs:
                    if isinstance(outputs[each], torch.Tensor):
                        derail = outputs[each].detach().cpu().numpy().squeeze().squeeze()
                        outputs[each] = derail
                    elif isinstance(outputs[each],list):
                        derail = outputs[each][0]
                        #print(derail)
                        #print(type(derail))
                        outputs[each] = derail
                # print(outputs)
                self.predictor.predictions_to_labeled_instances(new_instances[0], outputs)
                # print("after index: ",new_instances[0][target_field]._indexed_tokens)
                # logits = model_output["label_logits"].detach().cpu().numpy()[0]
                # new_label = numpy.argmax(logits)
                # print("label:", new_label, "logits:",logits)
                # print("-------------------------------------")
                # if new_label!=label:
                #     break
                label_change = False
                # if "tags" not in new_instances[0]:
                if "tags" in new_instances[0]:
                    cur_label_list = new_instances[0]["tags"].__dict__["field_list"]
                    print(cur_label_list)
                    cur_tag_dict = defaultdict(int)
                    cur_tag_tok = ''
                    for label in cur_label_list:
                        if label.label != "O":
                            cur_tag_dict[label.label] += 1
                            cur_tag_tok = cur_tag_tok + label.label
                    print(cur_tag_dict)
                    equal = (cur_tag_dict == tag_dict) and (cur_tag_tok==tag_tok)
                    if not equal:
                        break
                else:
                    for field in check_fields:
                        print(field)
                        # print(super(IndexField,new_instances[0][field]).__eq__(check_list[field]))
                        if field in new_instances[0].fields:
                            equal = new_instances[0][field].__eq__(check_list[field])
                            print(equal)
                            print(new_instances[0][field],check_list[field])
                        else:
                            equal = outputs[field] == check_list[field]
                            print(equal)
                            print(outputs[field],check_list[field])
                        if (not equal):
                            label_change = True
                            break
                if label_change:
                    break
                
            final_tokens.append(last_tokens)
        print(final_tokens)

        # print('\n')
        # print(check_fields)
        # print(new_instances[0]['label'].label)        
        return sanitize({"final": final_tokens,"original": original, "label": new_instances[0]['label'].label})
    def hotflip_attack(self,grad, embedding_matrix, adv_token_idx):    
        """
        TODO
        """
        grad = torch.from_numpy(grad)
        print(grad.shape)
        print(embedding_matrix.shape)
        embedding_matrix = embedding_matrix.cpu()    
        word_embeds = torch.nn.functional.embedding(torch.LongTensor([adv_token_idx]), embedding_matrix).detach().unsqueeze(0)
        grad = grad.unsqueeze(0).unsqueeze(0)  
        print(grad.shape)      
        new_embed_dot_grad = torch.einsum("bij,kj->bik", (grad, embedding_matrix))        
        prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, word_embeds)).unsqueeze(-1)         
        neg_dir_dot_grad = -1 * (prev_embed_dot_grad - new_embed_dot_grad)            
        score_at_each_step, best_at_each_step = neg_dir_dot_grad.max(2)                    
        return best_at_each_step[0]
