'''
The sum of the squares of the first ten natural numbers is,

1^2 + 2^2 + ... + 10^2 = 385

The square of the sum of the first ten natural numbers is,

(1 + 2 + ... + 10)^22 = 552 = 3025

Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is 3025 - 385 = 2640.

Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.
'''

def sum_square_minus_square_sum_difference(n):
    ''' solution to the module challenge '''
    your_code_here = 0


def test_sum_square_minus_square_sum_difference():
    assert 2640 == sum_square_minus_square_sum_difference(10)
    assert 25164150 == sum_square_minus_square_sum_difference(100)

