import unittest

from word_level import WordSparse


class TestWordSparse(unittest.TestCase):

    def test_start_chars(self):
        ws = WordSparse(num_chars=1)
        self.assertEqual(ws.start_chars('hello'), 'h')
        self.assertEqual(ws.start_chars('h'), 'h')
        ws = WordSparse(num_chars=2)
        self.assertEqual(ws.start_chars('hello'), 'he')
        self.assertEqual(ws.start_chars('he'), 'he')
        self.assertEqual(ws.start_chars('a'), 'a ')
        ws = WordSparse(num_chars=3)
        self.assertEqual(ws.start_chars('hel'), 'hel')
        self.assertEqual(ws.start_chars('hello'), 'hel')
        self.assertEqual(ws.start_chars('a'), 'a  ')

    def test_end_chars(self):
        ws = WordSparse(num_chars=1)
        self.assertEqual(ws.end_chars('hello'), 'o')
        self.assertEqual(ws.end_chars('h'), 'h')
        ws = WordSparse(num_chars=2)
        self.assertEqual(ws.end_chars('hello'), 'lo')
        self.assertEqual(ws.end_chars('he'), 'he')
        self.assertEqual(ws.end_chars('a'), ' a')
        ws = WordSparse(num_chars=3)
        self.assertEqual(ws.end_chars('hel'), 'hel')
        self.assertEqual(ws.end_chars('hello'), 'llo')
        self.assertEqual(ws.end_chars('a'), '  a')

    def test_preprocess(self):
        ws = WordSparse(num_chars=2)
        ws.preprocess('hello')
        self.assertTrue('he' in ws.start_char2id)
        self.assertTrue('lo' in ws.end_char2id)
        self.assertEqual(len(ws.start_char2id), 2)
        self.assertEqual(len(ws.end_char2id), 2)
        self.assertEqual(ws.start_char2id['he'], 1)
        self.assertEqual(ws.end_char2id['lo'], 1)
        ws.preprocess('world')
        self.assertTrue('wo' in ws.start_char2id)
        self.assertTrue('ld' in ws.end_char2id)
        self.assertEqual(len(ws.start_char2id), 3)
        self.assertEqual(len(ws.end_char2id), 3)
        self.assertEqual(ws.start_char2id['wo'], 2)
        self.assertEqual(ws.end_char2id['ld'], 2)

    def test_char_index(self):
        ws = WordSparse(num_chars=2)
        # should be unknown
        self.assertEqual(ws.start_char_index('he'), 0)
        self.assertEqual(ws.end_char_index('lo'), 0)

        ws.preprocess('hello')
        # should be known
        self.assertEqual(ws.start_char_index('he'), 1)
        self.assertEqual(ws.end_char_index('lo'), 1)

        # should be unknown
        self.assertEqual(ws.start_char_index('wo'), 0)
        self.assertEqual(ws.end_char_index('ld'), 0)

        ws.preprocess('world')
        # should be known
        self.assertEqual(ws.start_char_index('wo'), 2)
        self.assertEqual(ws.end_char_index('ld'), 2)

    #########################
    # static methods
    #########################

    def test_is_capital(self):
        self.assertTrue(WordSparse.is_capital('Hello'))
        self.assertTrue(WordSparse.is_capital('World'))
        self.assertFalse(WordSparse.is_capital('hi'))
        self.assertFalse(WordSparse.is_capital('there'))

    def test_trailing_s(self):
        self.assertTrue(WordSparse.trailing_s('Boogers'))
        self.assertTrue(WordSparse.trailing_s('lollipops'))
        self.assertFalse(WordSparse.trailing_s('Hello'))
        self.assertFalse(WordSparse.trailing_s('world'))

    def test_starting_vowel(self):
        self.assertTrue(WordSparse.starting_vowel('Around'))
        self.assertFalse(WordSparse.starting_vowel('the'))
        self.assertFalse(WordSparse.starting_vowel('world'))
        self.assertTrue(WordSparse.starting_vowel('animals'))
        self.assertTrue(WordSparse.starting_vowel('are'))
        self.assertFalse(WordSparse.starting_vowel('laughing'))

    def test_contains_num(self):
        self.assertTrue(WordSparse.contains_num('g2g'))
        self.assertFalse(WordSparse.contains_num('lolcats'))

    def test_contains_sym(self):
        self.assertTrue(WordSparse.contains_sym('hey!'))
        self.assertTrue(WordSparse.contains_sym('huh?'))
        self.assertTrue(WordSparse.contains_sym('n\'t'))
        self.assertTrue(WordSparse.contains_sym('co-operate'))
        self.assertFalse(WordSparse.contains_sym('hi'))
        self.assertFalse(WordSparse.contains_sym('there'))

    def test_only_sym(self):
        self.assertTrue(WordSparse.only_sym('!!!'))
        self.assertTrue(WordSparse.only_sym('$'))
        self.assertTrue(WordSparse.only_sym('%'))
        self.assertTrue(WordSparse.only_sym('~'))
        self.assertTrue(WordSparse.only_sym('##'))
        self.assertTrue(WordSparse.only_sym('?'))
        self.assertTrue(WordSparse.only_sym('??'))
        self.assertTrue(WordSparse.only_sym('...'))
        self.assertTrue(WordSparse.only_sym('--'))
        self.assertTrue(WordSparse.only_sym('='))
        self.assertTrue(WordSparse.only_sym('+'))
        self.assertFalse(WordSparse.only_sym('huh?'))
        self.assertFalse(WordSparse.only_sym('1+1=2'))
        self.assertFalse(WordSparse.only_sym('=v='))
        self.assertFalse(WordSparse.only_sym(':+0'))




if __name__ == '__main__':
    unittest.main()
