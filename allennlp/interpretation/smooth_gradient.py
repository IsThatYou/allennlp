import torch
from typing import List, Dict 
import numpy
from allennlp.common.util import JsonDict, sanitize, normalize_by_total_score
from allennlp.interpretation import Interpreter
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data import Instance

@Interpreter.register('smooth-gradient-interpreter')
class SmoothGradient(Interpreter):
  def __init__(self, predictor):
    super().__init__(predictor)

  def interpret_from_json(self, inputs: JsonDict) -> JsonDict:    
    labeled_instances = self.predictor.inputs_to_labeled_instances(inputs)

    instances_with_grads = dict()
    for idx, instance in enumerate(labeled_instances):      
      grads = self.smooth_grads(instance)      
      self._post_process(grads)
      instances_with_grads['instance_' + str(idx + 1)] = grads

    return sanitize(instances_with_grads)

  def _post_process(self, grads: Dict[str, numpy.ndarray]) -> None:    
    for key, grad in grads.items():      
      emb_grad = numpy.sum(grad, axis=1)
      normalized_grad = normalize_by_total_score(emb_grad)      
      grads[key] = normalized_grad 

  def _register_forward_hook(self, stdev: int):  
    def forward_hook(module, input, output):
      print("OUTPUT BEFORE")
      print("-------------")
      print(output)

      # sample random noise
      noise = torch.randn(output.shape).to(output.device) * (stdev * (output.detach().max() - output.detach().min()))
      
      # Change the embedding      
      output.add_(noise)
      
      print()
      print("OUTPUT AFTER")
      print("------------")
      print(output)
    
    # Register the hook
    handle = None
    for module in self.predictor._model.modules():
        if isinstance(module, TextFieldEmbedder):
            handle = module.register_forward_hook(forward_hook)

    return handle 

  def smooth_grads(self, instance: Instance) -> Dict[str, numpy.ndarray]:
    stdev = 0.01 
    num_samples = 25    
    total_gradients = None

    for i in range(num_samples):   
        # Define forward hook       
        handle = self._register_forward_hook(stdev)

        # Get gradients 
        grads = self.predictor.get_gradients([instance])[0]

        # Remove the hook
        handle.remove() 
        
        # Sum
        if total_gradients is None:
          total_gradients = grads
        else:
          for key in grads.keys():
            total_gradients[key] += grads[key]        

    # Average out
    for key in total_gradients.keys():
      total_gradients[key] /= num_samples
    
    return total_gradients