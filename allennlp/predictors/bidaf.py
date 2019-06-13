from typing import Dict, List
from overrides import overrides
import numpy
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import IndexField, TextField, ListField, LabelField, SpanField, SequenceLabelField
from allennlp.data.tokenizers import Token

@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage" : passage, "question" : question})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)

    @overrides
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        """
        TODO
        """
        #span_start = np.argmax(outputs['span_start_logits'])
        #span_end = np.argmax(outputs['span_end_logits'])
        #print(instance)
        #print(outputs)
        if 'best_span' in outputs: 
            span_ind = outputs['best_span']
            seq = instance["passage"]
            instance.add_field('span_start', IndexField(int(span_ind[0]),seq))
            instance.add_field('span_end', IndexField(int(span_ind[1]),seq))
        # This is for NAQANet model 
        elif 'answer' in outputs:
            #print(outputs)
            answer_type = outputs['answer']['answer_type']
            # When the problem is a counting problem
            if answer_type == 'count':
                field = ListField([LabelField(int(outputs['answer']['count']), skip_indexing=True)])
                instance.add_field('answer_as_counts', field)

            # When the answer is in the passage
            elif answer_type == 'passage_span':
                span = outputs['answer']['spans'][0]

                # NOTE
                # Below is the way the datasetreader gets the span, which returns
                # multiple positions if the answer appears multiple times 
                # answer_texts = [outputs['answer']['value']]
                # tokenized_answer_texts = []
                # for answer_text in answer_texts:
                #     answer_tokens = self._dataset_reader._tokenizer.tokenize(answer_text)
                #     answer_tokens = split_tokens_by_hyphen(answer_tokens)
                #     tokenized_answer_texts.append(' '.join(token.text for token in answer_tokens))
                # print('tokenized answer', tokenized_answer_texts)
                # real_span = self._dataset_reader.find_valid_spans(instance['passage'].tokens, tokenized_answer_texts)
                # print('real span', real_span)
                
                # Convert character span indices into word span indices
                word_span_start = None
                word_span_end = None
                for idx, offset in enumerate(instance['metadata'].metadata['passage_token_offsets']):
                    if offset[0] == span[0]:
                        word_span_start = idx
                    if offset[1] == span[1]:
                        word_span_end = idx 
                
                field = ListField([SpanField(word_span_start, word_span_end, instance['passage'])])
                instance.add_field('answer_as_passage_spans', field)

            # When the answer is an arithmetic calculation
            elif answer_type == 'arithmetic':
                # THIS SEEMS TO WORK, BUT NEEDS TESTING

                # These are basically the different numbers in the passage
                # the model encounters 
                sequence_labels = outputs['answer']['numbers']
                print('numbers length', len(sequence_labels))

                # NOTE
                # outputs['answer']['numbers'] does not include padding (like metadata)
                # do we need it?
                numbers_as_tokens = [Token(str(number['value'])) for number in outputs['answer']['numbers']]
                # hack copied from line 232 in drop.py
                numbers_as_tokens.append(Token('0'))
                numbers_in_passage_field = TextField(numbers_as_tokens, self._dataset_reader._token_indexers)

                # NOTE
                # Based on ``find_valid_add_sub_expressions``, it seems
                # like negative signs are 2 instead of -1
                # the numbers in the passage are given signs, that's what 
                # we are labeling here 
                labels = []
                for label in sequence_labels:
                    if label['sign'] == -1:
                        labels.append(2)
                    else:
                        labels.append(label['sign'])
                labels.append(0) # IS THIS CORRECT? (seems like it, 0 stands for "not included")

                field = ListField([SequenceLabelField(labels, numbers_in_passage_field)])
                instance.add_field('answer_as_add_sub_expressions', field)

            # When the answer is in the question
            elif answer_type == 'question_span':
                span = outputs['answer']['spans'][0]

                # Convert character span indices into word span indices
                word_span_start = None
                word_span_end = None
                for idx, offset in enumerate(instance['metadata'].metadata['question_token_offsets']):
                    if offset[0] == span[0]:
                        word_span_start = idx
                    if offset[1] == span[1]:
                        word_span_end = idx 

                field = ListField([SpanField(word_span_start, word_span_end, instance['question'])])
                instance.add_field('answer_as_question_spans', field)

        # print("modified instance", instance)
        return [instance]
