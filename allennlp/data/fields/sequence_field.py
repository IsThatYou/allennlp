from allennlp.data.fields.field import DataArray, Field


class SequenceField(Field[DataArray]):
    """
    A ``SequenceField`` represents a sequence of things.  This class just adds a method onto
    ``Field``: :func:`sequence_length`.  It exists so that ``SequenceLabelField``, ``IndexField`` and other
    similar ``Fields`` can have a single type to require, with a consistent API, whether they are
    pointing to words in a ``TextField``, items in a ``ListField``, or something else.
    """
    def sequence_length(self) -> int:
        """
        How many elements are there in this sequence?
        """
        raise NotImplementedError

    def empty_field(self) -> 'SequenceField':
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        print(self.__dict__)
        if isinstance(other, other.__class__):
            for each in self.__dict__:
                if isinstance(self.__dict__[each],list):
                    n = len(self.__dict__[each])
                    m = len(other.__dict__[each])
                    if n!=m:
                        return False
                    for i in range(n):
                        if self.__dict__[each][i] != other.__dict__[each][i]:
                            return False
                else:
                    if self.__dict__[each] != other.__dict__[each]:
                        return False
            return True
        else:
            return id(self) == id(other)
        # Otherwise it is not implmented
        return NotImplemented