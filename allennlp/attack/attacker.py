from allennlp.common import Registrable
from allennlp.common.util import JsonDict

class Attacker(Registrable):
	def __init__(self, predictor):
		self.predictor = predictor
	def attack_from_json(self, inputs:JsonDict):
		'''
		TODO
		'''
		raise NotImplementedError("you should implement this if you want to do model attack")
