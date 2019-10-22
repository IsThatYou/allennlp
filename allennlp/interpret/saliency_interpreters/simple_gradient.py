import math
from typing import List
import numpy
import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.nn import util
@SaliencyInterpreter.register("simple-gradient")
class SimpleGradient(SaliencyInterpreter):
    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Interprets the model's prediction for inputs.  Gets the gradients of the loss with respect
        to the input and returns those gradients normalized and sanitized.
        """
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        # List of embedding inputs, used for multiplying gradient by the input for normalization
        #       embeddings_list: List[numpy.ndarray] = []​
        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Hook used for saving embeddings
            handle = self._register_forward_hook(embeddings_list)
            grads = self.predictor.get_gradients([instance])[0]
            handle.remove()

            # Gradients come back in the reverse order that they were sent into the network
            embeddings_list.reverse()
            for key, grad in grads.items():
                # Get number at the end of every gradient key (they look like grad_input_[int],
                # we're getting this [int] part and subtracting 1 for zero-based indexing).
                # This is then used as an index into the reversed input array to match up the
                # gradient and its respective embedding.
                input_idx = int(key[-1]) - 1
                # The [0] here is undo-ing the batching that happens in get_gradients.
                emb_grad = numpy.sum(grad[0] * embeddings_list[input_idx], axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def _register_forward_hook(self, embeddings_list: List):
        """
        Finds all of the TextFieldEmbedders, and registers a forward hook onto them. When forward()
        is called, embeddings_list is filled with the embedding values. This is necessary because
        our normalization scheme multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inputs, output):
            embeddings_list.append(output.squeeze(0).clone().detach().numpy())

        embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = embedding_layer.register_forward_hook(forward_hook)

        return handle
  

    def saliency_interpret_from_instances(self, labeled_instances, embedding_operator, normalization,normalization2="l1_norm",do_softmax=False) -> JsonDict:
        # Get raw gradients and outputs
        grads, outputs = self.predictor.get_gradients(labeled_instances)

        final_loss = torch.zeros(grads["grad_input_1"].shape[1])
        rank = None
        joe_bob_position = 0 # TODO, hardcoded position
        softmax = torch.nn.Softmax(dim=0)
        # we only handle when we have 1 input at the moment, so this loop does nothing
        for key, grad in grads.items():
            # grads_summed_across_batch = torch.sum(grad, axis=0)
            for idx, gradient in enumerate(grad):
                # Get rid of embedding dimension
                summed_across_embedding_dim = None 
                if embedding_operator == "dot_product":
                    batch_tokens = labeled_instances[idx].fields['tokens']
                    batch_tokens = batch_tokens.as_tensor(batch_tokens.get_padding_lengths())
                    embeddings = self.predictor._model._text_field_embedder(batch_tokens)
                    embeddings = embeddings.squeeze(0).transpose(1,0)
                    summed_across_embedding_dim = torch.diag(torch.mm(gradient, embeddings))
                elif embedding_operator == "l2_norm":
                    summed_across_embedding_dim = torch.norm(gradient, dim=1)

                # Normalize the gradients 
                normalized_grads = None
                if normalization == "l2_norm":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim)
                elif normalization == "l1_norm":
                    normalized_grads = summed_across_embedding_dim / torch.norm(summed_across_embedding_dim, p=1)

                # Note we use absolute value of grad here because we only care about magnitude
                temp = [(idx, numpy.absolute(grad)) for idx, grad in enumerate(normalized_grads.detach().numpy())]
                temp.sort(key=lambda t: t[1], reverse=True)
                rank = [i for i, (idx, grad) in enumerate(temp) if idx == joe_bob_position][0]

                if normalization2 == "l2_norm":
                    normalized_grads = normalized_grads**2
                elif normalization2 == "l1_norm":
                    normalized_grads = torch.abs(normalized_grads)

                normalized_grads = torch.nn.utils.rnn.pad_sequence([final_loss, normalized_grads]).transpose(1, 0)[1]

                final_loss += normalized_grads
        final_loss /= grads['grad_input_1'].shape[0]
        # L1/L2 norm/sum, -> softmax
        if do_softmax:
            normalized_grads = softmax(normalized_grads)

        
        # print("finetuned loss", final_loss)
        final_loss = final_loss[joe_bob_position]
        final_loss.requires_grad_()
        return final_loss, rank
