from allennlp.attack import Attacker
class Pathologies(Attacker):
	def __init__(self, predictor):
		super().__init__(predictor)
	def attack_from_json(self, inputs:JsonDict):
		'''
		TODO
		'''
		pass
		