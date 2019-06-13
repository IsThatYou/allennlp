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
        if isinstance(other, other.__class__):
            for each in self.__dict__:
                if self.__dict__[each] != other.__dict__[each]:
                    return False
            return True
        else:
            return id(self) == id(other)
        # Otherwise it is not implmented
        return NotImplemented