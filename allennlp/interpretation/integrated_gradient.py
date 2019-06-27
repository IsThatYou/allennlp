from typing import List, Dict 

import numpy

from allennlp.common.util import JsonDict, sanitize, normalize_by_total_score
from allennlp.interpretation import Interpreter
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data import Instance


@Interpreter.register('integrated-gradients-interpreter')
class IntegratedGradient(Interpreter):
  def __init__(self, predictor):
    super().__init__(predictor)

  def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)  
    """

    # Get labeled instances
    labeled_instances = self.predictor.inputs_to_labeled_instances(inputs)

    instances_with_grads = dict()
    for idx, instance in enumerate(labeled_instances):
      print("INTEGRATED GRADS")
      print('----------------')
      grads = self.integrate_gradients(instance)
      print(grads)
      print()

      self._post_process(grads)

      instances_with_grads['instance_' + str(idx + 1)] = grads

    print("INSTANCES WITH GRADS")
    print("--------------------")
    print(instances_with_grads)
    return sanitize(instances_with_grads)

  def _post_process(self, grads: Dict[str, numpy.ndarray]) -> None:
    """
    Take the L2-norm across the embedding dimension and normalize. 
    """
    for key, grad in grads.items():
      # *** L2 STUFF ***
      # l2_grad = numpy.linalg.norm(grad, axis=1)
      # normalized_grad = normalize_by_total_score(l2_grad)

      # *** EMB STUFF ***
      emb_grad = numpy.sum(grad, axis=1)
      normalized_grad = normalize_by_total_score(emb_grad)
      print('NORMALIZED GRAD')
      print('---------------')
      print(normalized_grad.shape)
      grads[key] = normalized_grad 

  def _register_forward_hook(self, alpha: int, inputs: List):
    """
    Register a forward hook on the embedding layer which multiplies
    the output with the alpha parameter. 

    Notes
    -----
    When alpha is zero, we add the output to the inputs list passed in.
    This is done so that later we can use these inputs to elment-wise
    multiply them with the accumulated gradients. 
    """
    def forward_hook(module, input, output):
      print("OUTPUT BEFORE")
      print("-------------")
      print(output)

      # We need to save the inputs to multiply later, but only once
      # Order is important here, generalizable across models?
      if alpha == 0:
        inputs.append(output.squeeze(0).clone().detach().numpy())

      # Change the embedding
      output.mul_(alpha)

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

  def integrate_gradients(self, instance: Instance) -> Dict[str, numpy.ndarray]:
    """
    Returns integrated gradients for the given :class:`~allennlp.data.instance.Instance`,
    according to https://arxiv.org/abs/1703.01365. 
    """
    # Initialize Integrated Gradients
    ig_grads = None

    # List of Embedding inputs
    inputs = []

    # How to find best step size?
    steps = 10

    # We don't include the endpoint since we're doing a left point 
    # approximation to the integral 
    for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):

      # Define forward hook
      handle = self._register_forward_hook(alpha, inputs)

      # Get gradients 
      grads = self.predictor.get_gradients([instance])[0]

      # Remove the hook
      handle.remove() 

      # Add to IG gradients 
      if ig_grads is None:
        ig_grads = grads
      else:
        for key in grads.keys():
          ig_grads[key] += grads[key]

    # Average out
    for key in ig_grads.keys():
      ig_grads[key] *= 1/steps

    print("IG GRADS BEFORE INPUT MULTIPLY")
    print("------------------------------")
    print(ig_grads)

    # It seems like the gradients come back in the reverse order that 
    # they were sent into the network
    inputs.reverse()
    print("INPUTS TO MULTIPLY")
    print("------------------")
    print(inputs)

    # Element-Wise multiply input 
    for idx, iput in enumerate(inputs):
      key = "grad_input_" + str(idx + 1)
      assert ig_grads[key].shape == iput.shape, 'Error, shapes not the same!'
      ig_grads[key] *= iput

    return ig_grads  
