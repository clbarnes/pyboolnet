def prod(numbers):
    """
    Find the product of a sequence

    :param numbers: Sequence of numbers
    :return: Their product
    """
    ret = 1
    for number in numbers:
        ret *= number

    return ret