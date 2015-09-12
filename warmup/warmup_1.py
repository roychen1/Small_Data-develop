'''
A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 * 99.
Find the largest palindrome made from the product of two 3-digit numbers.
'''

def is_a_palindrome(x):
    ''' tests if something's string representation is a palendrome '''
    your_code_here = 0

def largest_palindrome_from_product_of_2_n_digit_numbers(n):
    ''' solution to the module challenge '''
    your_code_here = 0


def test_is_a_palindrome():
    assert is_a_palindrome(9009)
    assert is_a_palindrome(20102)
    assert is_a_palindrome(123454321)
    assert not is_a_palindrome(9000)
    assert not is_a_palindrome(12345)
    assert not is_a_palindrome(543)

def test_largest_palindrome_from_product_of_2_n_digit_numbers():
    assert 9009 == largest_palindrome_from_product_of_2_n_digit_numbers(2)
    assert 906609 == largest_palindrome_from_product_of_2_n_digit_numbers(3)

