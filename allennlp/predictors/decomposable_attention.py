from typing import Dict, List, Set
import numpy as np

from overrides import overrides
from allennlp.common.util import JsonDict, sanitize 
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields import LabelField, MetadataField
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.tokenizers import Token
from allennlp.nn import util
import torch

@Predictor.register('textual-entailment')
class DecomposableAttentionPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.

        Parameters
        ----------
        premise : ``str``
            A passage representing what is assumed to be true.

        hypothesis : ``str``
            A sentence that may be entailed by the premise.

        Returns
        -------
        A dictionary where the key "label_probs" determines the probabilities of each of
        [entailment, contradiction, neutral].
        """
        return self.predict_json({"premise" : premise, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)

    @overrides
    def predictions_to_labels(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> List[Instance]:
        """
        TODO
        """
        label = np.argmax(outputs['label_logits'])
        instance.add_field('label', LabelField(int(label), skip_indexing=True))
        #instance.add_field('label_logits', MetadataField(outputs['label_logits']))
        return [instance]

    @overrides
    def attack_from_json(self, inputs: JsonDict, target_field: str = "hypothesis", ignore_tokens:List[str] = ["@@NULL@@"]) -> Dict[str, np.ndarray]:
        """
        TODO
        """
        # Left one out code
        #return self.LOO(inputs)
        ########################
        # hot flip code

        #return self.hotflip(inputs,target_field,ignore_tokens)
        ########################
        
        new_instances = self.get_model_predictions(inputs)
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

          grads,outputs = self.get_gradients(new_instances)
          model_output = self._model.decode(outputs)
          logits = model_output["label_logits"].detach().cpu().numpy()[0]
          new_label = np.argmax(logits)
          print("label:", new_label, "logits:",logits)
          print("------------------------")
          if (new_label!=label):
            #print(last_tokens)
            break
          #print(grads.keys())
          last_tokens = list(new_instances[0][target_field].tokens)
          new_instances = self.pathological_attack(grads["grad_input1"], new_instances, target_field, ignore_tokens)
          print(new_instances[0])

        print("final adv:", last_tokens," | label:",last_label," | logits:", last_logits)
        # TODO: return something else
        return sanitize({"final": last_tokens})
        
    def hotflip(self):
        new_instances = self.get_model_predictions(inputs)
        label = new_instances[0]["label"].label
        print("start label:",label)
        embedder = self._model._text_field_embedder._token_embedders["tokens"]
        vocab = self._model.vocab
        all_tokens = list(vocab._token_to_index["tokens"].keys())
        a = [x for x in vocab._index_to_token["tokens"].keys()]
        b = torch.LongTensor(a)
        b = b.unsqueeze(0)
        c = {"tokens":b}
        print("b= ",b.shape)

        for module in self._model.modules():
            if isinstance(module, BasicTextFieldEmbedder):
                M = module
        in_text_field = new_instances[0][target_field]._indexed_tokens
        padding_length = new_instances[0][target_field].get_padding_lengths()
        temp = new_instances[0][target_field].as_tensor(padding_length)
        print(temp)
        temp2 = embedder(temp["tokens"])
        print(temp2[0])
        embedding_matrix = M(c)
        embedding_matrix = embedding_matrix.squeeze()
        #print(embedding_matrix[476])
        print("embedding matrix shape =",embedding_matrix.shape)
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=embedding_matrix.shape[1], weight=embedding_matrix,trainable=False)
        #print(token_embedding)

        print(M._token_embedders["tokens"].weight.shape)
        tokens = new_instances[0][target_field].tokens
        print(tokens)
        indexed_tokens = in_text_field["tokens"]
        which_token = 1
        adv_token_idx = indexed_tokens[which_token]
        print(adv_token_idx)

        new_instances = self.get_model_predictions(inputs)
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

        while (True) :
          last_label = new_label
          last_logits = logits
          #new_instances[0].fields.pop('label_logits', None)

          grads,outputs = self.get_gradients(new_instances)
          model_output = self._model.decode(outputs)
          logits = model_output["label_logits"].detach().cpu().numpy()[0]
          new_label = np.argmax(logits)
          print("label:", new_label, "logits:",logits)
          print("------------------------")
          if (new_label!=label):
            #print(last_tokens)
            break
          #print(grads.keys())
          last_tokens = list(new_instances[0][target_field].tokens)
          new_instances = self.pathological_attack(grads["grad_input1"], new_instances, target_field, ignore_tokens)
          print(new_instances[0])

        print("final adv:", last_tokens," | label:",last_label," | logits:", last_logits)
        # TODO: return something else
        return sanitize({"final": last_tokens})


        # grads,outputs = self.get_gradients(new_instances)
        # model_output = self._model.decode(outputs)
        # grad = grads["grad_input1"]
        # ret = self.hotflip_attack(grad[which_token], embedding_matrix, adv_token_idx)
        # ret = ret.data[0].detach().cpu().item()
        # print(ret)
        # i2t = vocab._index_to_token["tokens"]
        # print(i2t[ret])
        # print(new_instances[0][target_field].tokens)
        # new_instances[0][target_field].tokens[which_token] = Token(i2t[ret])
        # new_instances[0].indexed = False
        # print(new_instances[0][target_field].tokens)

        # grads,outputs = self.get_gradients(new_instances)
        # model_output = self._model.decode(outputs)
        # logits = model_output["label_logits"].detach().cpu().numpy()[0]
        # new_label = np.argmax(logits)
        # print("label:", new_label, "logits:",logits)
        # return {"final":"a"}

    def beam_search(self, instances:List[Instance], label:int) -> List[Instance]:
        """
        TODO
        """
        grad_list = []
        instance_list = []
        length = len(instances[0]["hypothesis"])
        while length>1 :
            for i in range(len(instances)):
                cur_instance = instances[i]
                grads,outputs = self.get_gradients(new_instances)
                new_instance = self.pathological_attack(grads["grad_input1"], [cur_instance])
                outputs = self._model.forward_on_instances([new_instance])
                logits = outputs[0]['label_logits']
                new_label = np.argmax(logits)
                if (new_label == label):
                    word_list.append(new_instance)
        return instance_list

    def LOO(self,inputs: JsonDict) -> List:
        """
        TODO
        """
        instances = self.get_model_predictions(inputs)
        label = instances[0]["label"].label
        instance_list = []
        sentence_tensor = instances[0]["hypothesis"].tokens

        outputs = self._model.forward_on_instances(instances)
        logits = outputs[0]['label_logits']
        print("Original sentence:",sentence_tensor,"original confidence:",logits[label])
        for i in range(len(sentence_tensor)):
            cur = sentence_tensor[:]
            del cur[i]
            
            instances[0]["hypothesis"].tokens = cur
            instances[0].indexed = False
            outputs = self._model.forward_on_instances(instances)
            logits = outputs[0]['label_logits']
            print("LOO sentence:",cur,"confidence:",logits[label])

            instance_list.append({'sentence':cur,'confidence':logits[label]})
        print(instance_list)
        return instance_list


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