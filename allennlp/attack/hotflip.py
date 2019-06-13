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

@Attacker.register('hotflip')
class Hotflip(Attacker):
    def __init__(self, predictor):
        super().__init__(predictor)
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                M = module
        print(M)
        vocab = self.predictor._model.vocab
        self.vocab = vocab
        all_tokens = list(vocab._token_to_index["tokens"].keys())
        a = [x for x in vocab._index_to_token["tokens"].keys()]
        b = torch.LongTensor(a)
        b = b.unsqueeze(0)
        c = {"tokens":b}
        print("b = ",b.shape)
        embedding_matrix = M(c)
        embedding_matrix = embedding_matrix.squeeze()
        #print(embedding_matrix[476])
        print("embedding matrix shape =",embedding_matrix.shape)
        self.token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=embedding_matrix.shape[1], weight=embedding_matrix,trainable=False)


    def attack_from_json(self, inputs:JsonDict, target_field: str, gradient_index:str,ignore_tokens:List[str] = ["@@NULL@@"]):
        new_instances = self.predictor.inputs_to_labeled_instances(inputs)
        label = new_instances[0]["label"].label
        # embedder = self.predictor._model._text_field_embedder._token_embedders["tokens"]
        
        in_text_field = new_instances[0][target_field]._indexed_tokens
        original = new_instances[0][target_field].tokens
        print(original)
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

        new_instances = self.predictor.inputs_to_labeled_instances(inputs)
        
        # handling ignore tokens by creating a mask
        ignore_tokens = set(ignore_tokens)
        num_ignore_tokens = 0
        ignore_tokens_mask = [0]*len(new_instances[0][target_field].tokens)
        for idx,token in enumerate(new_instances[0][target_field].tokens):
            if str(token) in ignore_tokens:
                num_ignore_tokens += 1
                ignore_tokens_mask[idx] = 1
        print(ignore_tokens_mask)


        last_tokens = new_instances[0][target_field].tokens



        grads,outputs = self.predictor.get_gradients(new_instances)
        model_output = self.predictor._model.decode(outputs)
        print("start label:",label,"logits: ",model_output["label_logits"])
        n = len(indexed_tokens)
        while True:
            if which_token>=n:
                break
            grad = grads["grad_input_1"]
            num_of_words = grad.shape[0]
            grads_mag = []
            for i in range(num_of_words):
                norm = numpy.sqrt(grad[i].dot(grad[i]))
                grads_mag.append(norm)
            #print(grads_mag)
            which_token = numpy.argmax(grads_mag)

            
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
            model_output = self.predictor._model.decode(outputs)
            print("after index: ",new_instances[0][target_field]._indexed_tokens)
            logits = model_output["label_logits"].detach().cpu().numpy()[0]
            new_label = numpy.argmax(logits)
            print("label:", new_label, "logits:",logits)
            print("-------------------------------------")
            if new_label!=label:
                break
            #which_token+=1
        
        return sanitize({"final": last_tokens,"original": original})
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
