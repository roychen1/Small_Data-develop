#!/usr/bin/env python
''' primary entry point for the NLP pipeline '''


import collections
import itertools
import logging
import operator
import re
import os
import string
import sys
import hashlib
import copy

import nltk
import nose
import unittest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#sys.path.append('C:\Users\NDT567\Small_Data\source')

from settings import INPUT_FILE_PATH_SAMPLE as INPUT_FILE_PATH 
import settings




# numpy options
np.set_printoptions(linewidth=150)

# what is a random seed?  http://en.wikipedia.org/wiki/Random_seed
# why is it a good idea? http://stats.stackexchange.com/questions/121225/references-and-best-practices-for-setting-seeds-in-pseudo-random-number-generati
np.random.seed(42)

# logging configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


# module variables
UserSpeechFragment = collections.namedtuple('UserSpeechFragment', ['role', 'fragment_text']) #A named tuple containing who said what text
AGENT, CALLER = 'agent', 'caller' #The names given to the agent and the caller in the transcripts


def quiz_wiki_url_for_first_speech_recognition():
    ''' on the Speech Recognition Wikipedia page, look in the 'History'
        section, and find the first organization to investigate speech 
        recognition. Return that organization's wiki url.

        test with:
            nosetests -v run_project:test_quiz_wiki_url_for_first_speech_recognition '''
    return 'http://en.wikipedia.org/wiki/Bell_Labs'

def test_quiz_wiki_url_for_first_speech_recognition():
    assert '35a23e07d7b4ec4d76eb96b46857a05f115f07920b7dd81239c96931' == hashlib.sha224(quiz_wiki_url_for_first_speech_recognition()).hexdigest()


def get_lines_from_transcript_file(file_path=INPUT_FILE_PATH):
    ''' using the "with" statement, the "open" function,
        the INPUT_FILE_PATH variable defined above, and the "yield" keyword,
        return a generator that yields each line of the transcripts file.
        Make sure to .lower() each line - it'll be easier to process
        the text with no capital letters.

        Arguments:
        file_path: A string containing the filename of the file
            to read in. Defaults to the INPUT_FILE_PATH variable.

        Returns:
        A generator which returns the next line of the file each
           time its .next() function is called.
           
        Read the Green-Checked answer to this StackOverflow question:
            http://stackoverflow.com/questions/11555468/how-should-i-read-a-file-line-by-line-in-python
        More info on using the with statement:
            http://effbot.org/zone/python-with-statement.htm
        More info on yield statements:
            http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python

        test with:
            nosetests -v run_project:test_get_lines_from_transcript_file '''
    with open(file_path) as fp:
        for line in fp:
            yield line.lower()

     
class test_get_lines_from_transcript_file(object):
    def setup(self):
        self.test_filename = 'test_file.nogit'
        test_file = open(self.test_filename, 'wb')
        test_file.write('This is line 1.\nThis is line 2.\n')
        test_file.close()
        self.line_generator = get_lines_from_transcript_file(file_path=self.test_filename)
    def teardown(self):
        self.line_generator.close()
        os.remove(self.test_filename)
    def test_get_lines_from_transcript_file_line0(self):
        line0 = self.line_generator.next()
        assert line0 == 'this is line 1.\n'
    def test_get_lines_from_transcript_file_line1(self):
        line0 = self.line_generator.next()
        line1 = self.line_generator.next()
        assert line1 == 'this is line 2.\n'


def line_to_feature_dictionary(line):
    ''' Each call record has several fields, containing information
        about the call as well as the transcript. Edit this function
        to split a pipe (|) delimited call record up into its components,
        removing the leading and trailing whitespace around each field.

        Arguments:
        line: A single call record, with sections delimited
            by the pipe (|) character.

        Returns:
        call_dict: A dictionary of the call components,
            with the keys listed in the 'columns' list.

        Hint: use line.split('|'), and check out the string.strip()
            command. To make the output dictionary, you can start
            with an empty dictionary ({}), then iterate through
            the columns list adding key:value pairs to it.

            All remaining prior information regarding the data schema is available here:
                https://github.kdc.capitalone.com/BigDataAcademy/Small_Data/issues/24

        test with:
            nosetests -v run_project:test_line_to_feature_dictionary '''

    columns = ['audio_file_name',#Name of the audio file transcript
               'date',#Date of call
               'transcript',#The call text
               'hvs_or_core',#Where the call was routed
               'zero_or_one0',#dummy field?
               'zero_or_one1',#dummy field?
               'customer_net_promoter_score_response',#NPS score
               'blank_field']#dummy field due to trailing |
    d={}
    for i in range (len(line.split('|'))):
        d[columns[i]]=string.strip(line.split('|')[i])
    return d
    

def test_line_to_feature_dictionary():
    # dummy input data
    audio_file_name = 'cap1_core_2014-07-27-02-05-42_2950008_21_6040556861322767314_1_46.wav'
    date = '27jul14'
    transcript = 'speaker1 (agent): thank you for calling capital one.'
    hvs_or_core = 'hvs'
    zero_or_one0 = '0'
    zero_or_one1 = '1'
    customer_net_promoter_score_response = ''
    blank_field = ''

    input_row = [audio_file_name, date, transcript, hvs_or_core,
                 zero_or_one0, zero_or_one1, customer_net_promoter_score_response, blank_field]

    input_line = ' |'.join(input_row)

    expected_output = {'audio_file_name': audio_file_name,
                       'date': date,
                       'transcript': transcript,
                        'hvs_or_core': hvs_or_core,
                        'zero_or_one0': zero_or_one0,
                        'zero_or_one1': zero_or_one1,
                        'customer_net_promoter_score_response': customer_net_promoter_score_response,
                        'blank_field': blank_field}


    feature_dict = line_to_feature_dictionary(input_line)

    assert sorted(feature_dict.items()) == sorted(expected_output.items())

    
def split_transcript_by_speaker(transcript):
    '''
    The transcript indicates that the speaker has changed
    with the text 'speakerN (caller/agent):'. This is great,
    because we want to split the wall of text that is the
    transcript into lists of the pieces of conversation,
    split both by caller vs. agent as well as into coherent thoughts
    (i.e. blocks of uninterrupted speaking).

    We'll do the latter splitting here, which boils down to
    parsing through a single string, finding any instances of 'speakerN',
    and splitting the string (while removing the 'speakerN' text
    since it's not actually a part of the conversation). Since we'll
    want to split based on who the speaker is later on,
    we'll leave the (caller/agent) part in the transcript for now.

    Arguments:
    transcript: A string that is a single call center transcript.

    Returns:
    A list of strings, one string for each time the speaker changes.
        To be safe, any empty strings ('' or None) in the list should
        be filtered out of the list.

    Hint: You could write a custom parser for the text to find instances
        of'speaker1' and 'speaker2', which is a totally acceptable way to
        solve this problem. However, if that sounds like a lot of work,
        look into using regular expressions - see
        https://docs.python.org/2/howto/regex.html for details on how
        to make a regular expression as well as how to use it to
        split text.

    Test with:
        nosetests -v run_project:test_split_transcript_by_speaker
    '''
    
    a=[item for item in re.split('speaker. ', transcript)]
    return filter(None, a)



def test_split_transcript_by_speaker():
    input_transcript = 'speaker1 (caller): hello. speaker2 (agent): thank you for calling capital one.'
    expected_output = ['(caller): hello. ', '(agent): thank you for calling capital one.']
    assert expected_output == split_transcript_by_speaker(input_transcript)


def transcript_fragment_to_speaker_text_tuple(transcript_fragment):
    '''
    With the transcript broken up by speaker, the next thing we
    want to do is identify who the speaker is. This function takes
    in a single entry of the list returned by split_transcript_by_speaker
    (a 'transcript fragment') and figures out a) who the speaker was
    and b) what they said.

    Arguments:
    transcript_fragment: a string containing text spoken by a single
        party of the conversation (either the agent or the caller),
        with an identification of the party given in parentheses before
        the text. For example: '(caller): I am the person calling in. '

    Returns:
    identified_fragment: An instance of the UserSpeechFragment
        named tuple (defined at the top of this file, in the module
        variables section).

    Hint 1: You may not have encountered named tuples before. Don't
        worry! They're not too complex. You can think of them as
        regular tuples, but instead of accessing their values
        by a numerical index:
        test_tuple = ('hi',"I'm",'a','tuple')
        print test_tuple[2] -> prints 'a'

        You can instead access their values by string:
        test_namedtuple = UserSpeechFragment(role='caller',fragment_text='hello.')
        print test_namedtuple.role -> prints 'caller'

        Note that named tuples retain all the functionality of regular tuples:
        print test_namedtuple[1] -> prints 'hello.'

        For more details about named tuples, check out these links:
        http://pymotw.com/2/collections/namedtuple.html
        http://sys-exit.blogspot.in/2013/12/python-named-tuples.html

    Hint 2: Like for split_transcript_by_speaker, while you could write
        a full text parser for this specific purpose yourself, a regular
        expression is really the proper tool for this job. The needed
        regular expression is much more complicated than in
        split_transcript_by_speaker, however, so we'll give it to you:
        regex_string = r'[(](\w+)[)]:\s(.*)'

        However, we're not going to tell you exactly how to use it
        (that would spoil the fun!). Take a close look at the Python
        regular expression documentation
        (https://docs.python.org/2/howto/regex.html), especially the
        section on 'Grouping'.
        
    Test with:
        nosetests -v run_project:test_transcript_fragment_to_speaker_text_tuple
    '''
    UserSpeechFragment = collections.namedtuple('UserSpeechFragment', ['role', 'fragment_text']) 
    aa=[]    
    for l in transcript_fragment:
        #aa=filter(None,re.split(r'[(](\w+)[)]:\s(.*)',l)) 
        if len(filter(None,re.split(r'[(](\w+)[)]:\s(.*)',l)))>1:
            aa=aa+ [UserSpeechFragment(role=filter(None,re.split(r'[(](\w+)[)]:\s(.*)',l))[0], fragment_text=filter(None,re.split(r'[(](\w+)[)]:\s(.*)',l))[1].strip())]
    return aa
    
def test_transcript_fragment_to_speaker_text_tuple():
    transcript_fragment = '({0:s}): hello. '.format(CALLER)
    expected_output = (CALLER, 'hello.')
    assert expected_output == transcript_fragment_to_speaker_text_tuple(transcript_fragment)


def consolidate_speaker_runs(assigned_transcripts):
    '''
    Take a list of transcript fragments (each one a UserSpeechFragment)
    and merge any occurances where the text transcription software has split
    up a single block of speech into multiple blocks. For instance,
    if the transcript reads 'speaker1 (agent): Thank you for. speaker1 (agent): Calling
    Capital One. speaker2 (caller): You're welcome.' then our processing pipeline will produce a list of UserSpeechFragments like this:
    [UserSpeechFragment(role='agent',fragment_text='thank you for.'),
     UserSpeechFragment(role='agent',fragment_text='calling capital one.'),
     UserSpeechFragment(role='caller',fragment_text='you're welcome.')
    ]

    When we'd really like:
    [UserSpeechFragment(role='agent',fragment_text='thank you for. calling capital one.'),
     UserSpeechFragment(role='caller',fragment_text='you're welcome.')
    ]

    (The messed up punctuation is annoying, but not something to worry about.)

    Arguments:
    assigned_transcripts: a list of UserSpeechFragments to be consolidated

    Returns:
    consolidated_transcripts: a list of UserSpeechFragments, identical to
        the input list *except* in that multiple consecutive fragments
        with the same role have been merged into a single fragment.
    
    Test with:
        nosetests -v run_project:test_consolidate_speaker_runs
    '''

    UserSpeechFragment = collections.namedtuple('UserSpeechFragment', ['role', 'fragment_text']) 
    dd=[]
    i=0
    while i <len(assigned_transcripts):
        d=assigned_transcripts[i].fragment_text
        while i <>len(assigned_transcripts)-1 and assigned_transcripts[i].role==assigned_transcripts[i+1].role:
            d=d+' ' + assigned_transcripts[i+1].fragment_text
            i=i+1    
        dd=dd+[UserSpeechFragment(assigned_transcripts[i].role, d)]
        i=i+1
    return dd

def test_consolidate_speaker_runs():
    input_collection = [UserSpeechFragment(CALLER, 'hello'),
                        UserSpeechFragment(CALLER, 'world'),
                        UserSpeechFragment(AGENT, 'thank you for calling capital one.'),
                        UserSpeechFragment(CALLER, "You're welcome.")]
    input_collection_copy = copy.deepcopy(input_collection)

    expected_output = [UserSpeechFragment(CALLER, 'hello world'),
                       UserSpeechFragment(AGENT, 'thank you for calling capital one.'),
                       UserSpeechFragment(CALLER, "You're welcome.")]

    assert expected_output == consolidate_speaker_runs(input_collection)

    assert input_collection ==  input_collection_copy #see below if you fail
    # https://github.kdc.capitalone.com/pages/BigDataAcademy/Python_Foundations_I/modules/Datatypes.html#mutability


def transcript_to_dialogue_collection(transcript):
    '''
    This function acts as a wrapper around the 'split_transcript_by_speaker',
    'transcript_fragment_to_speaker_text_tuple', and
    'consolidate_speaker_runs' functions. Starting from a transcript,
    do the following:
    1. Split the transcript into a list of strings, one for each time the
        transcription software identifies a speaker.
    2. For each string in the list, convert the raw string into a
        UserSpeechFragment named tuple.
    3. Remove any list elements that are None (i.e. places where
        the conversion to UserSpeechFragment failed).
    4. Consolidate the filtered list of UserSpeechFragments for
        instances where the same speaker spoke twice (or more) times
        in a row
    5. Return this consolidated list of UserSpeechFragments.

    Arguments:
    transcript: A string representing a full call transcript (the
        same as you would input to 'transcript_fragment_to_speaker_text_tuple')

    Returns:
    transcript_text_collection: A list of UserSpeechFragments, generated
        as listed in the steps above.
        
    Test with:
        nosetests -v run_project:test_transcript_to_dialogue_collection
    '''
    transcript_fragment=split_transcript_by_speaker(transcript)
    assigned_transcripts=transcript_fragment_to_speaker_text_tuple(transcript_fragment)
    return consolidate_speaker_runs(assigned_transcripts)

def test_transcript_to_dialogue_collection():
    transcript = 'speaker1 ({0:s}): Hello. speaker1 ({0:s}): World. speaker2 ({1:s}): Thank you for calling Capital One.'.format(CALLER,AGENT)
    expected_output = [(CALLER, 'Hello. World.'), (AGENT, 'Thank you for calling Capital One.')]
    assert expected_output == transcript_to_dialogue_collection(transcript)


def dialogue_collection_pretty_print(dialogue_collection):
    '''
    This is just a helper function to make things print nicer.
    Don't worry about it :)
    '''
    lines = map(lambda t: '\t{}\t: {}'.format(*t), dialogue_collection)
    return '\n'.join(lines)



# a print_debugging_function
def print_out_munging_results():
    ''' take a small sample, the first few records in the transcript file
        perform the row-ification and dialogue collection transformation
        then print each fragment in each dialogue
        
        run with:
             python -c 'import run_project;run_project.print_out_munging_results()'
    '''
    lines = get_lines_from_transcript_file()
    lines = itertools.islice(lines, 10**3)
    for line in lines:
        row = line_to_feature_dictionary(line)
        dialogue_collection = transcript_to_dialogue_collection(row['transcript'])
        for fragment in dialogue_collection:
            print '\t', fragment
        print



def quiz_printed_value_of_logging_dot_ERROR():
    '''
    Open the python interpreter and import
    the logging module. Then run print logging.ERROR.

    Make this function return what the interpreter prints
    to the screen.

    test with:
        nosetests -v run_project:test_quiz_printed_value_of_logging_dot_ERROR
    '''
    return 40


def test_quiz_printed_value_of_logging_dot_ERROR():
    assert 'af91525e3568e15041ccfa1a7e58f2a8eac837b3c89f400e3f8aea0a' == hashlib.sha224(str(quiz_printed_value_of_logging_dot_ERROR())).hexdigest()



def dialogue_collection_to_string(dialogue_collection, role_filter=None):
    '''
    Convert a list of UserSpeechFragments back into a single string,
    with the fragments separated by spaces. The default behavior of
    this function is designed to take the fragmented transcript and
    just return the whole thing as a single string - almost like undoing
    all the work that we did in fragmenting the data, **EXCEPT THAT THE
    TEXT RELATING TO SPEAKER IDENTIFICATION IS GONE**. This is pretty
    important, since that text will only serve to make any kind of analysis
    we do worse - it's not part of the actual conversation, after all.

    But wait, there's more! When the role_filter argument is not set
    to None, this function will actually return a single string for
    *just* the person defined by the role_filter argument. So, for
    example, if you set role_filter='caller', you'll get back a
    single string which only has the text from the caller merged together.

    Arguments:
    dialogue_collection: A list of UserSpeechFragments, such as that
        returned by consolidate_speaker_runs.
    role_filter (default=None): if not None, tells the function to
        only return text spoken by the person in that role (e.g.
        'caller' or 'agent').

    Returns:
    collection_string: A single string representing the merged
        UserSpeechFragments, with each fragment separated by a space (' ').

    Hint: Google "join list of strings python" for help on merging the
        fragments together.
        
    Test with:
        nosetests -v run_project:test_dialogue_collection_to_string
    '''
    text=''
    if role_filter==None:
        for i in range(len(dialogue_collection)):
            text=text + dialogue_collection[i].fragment_text + ' '
    else:
        for i in range(len(dialogue_collection)):
            if dialogue_collection[i].role==role_filter:
                text=text + dialogue_collection[i].fragment_text + ' '
    return text.strip()
            
def get_agent_transcript(dialogue_collection):
    ''' syntactic sugar on top of dialogue_collection_to_string '''
    return dialogue_collection_to_string(dialogue_collection, role_filter=AGENT)

def get_caller_transcript(dialogue_collection):
    ''' syntactic sugar on top of dialogue_collection_to_string '''
    return dialogue_collection_to_string(dialogue_collection, role_filter=CALLER)
    
class test_dialogue_collection_to_string(object):
    def setup(self):
        self.fragment_collection = [UserSpeechFragment(AGENT, 'hello'),
                                    UserSpeechFragment(CALLER, 'hi'),
                                    UserSpeechFragment(AGENT, 'how is it going?'),
                                    UserSpeechFragment(CALLER, 'great!')]
    def is_string(self,obj):
        try:
            obj + ''
            return True
        except TypeError:
            return False
    def test_dialogue_collection_to_string_norole(self):
        expected_output = 'hello hi how is it going? great!'
        received_output = '[NOT A STRING]'
        if self.is_string(dialogue_collection_to_string(self.fragment_collection)):
            received_output = dialogue_collection_to_string(self.fragment_collection)
        assert expected_output == dialogue_collection_to_string(self.fragment_collection),\
           "Expected '{0:s}' from [{1:s}]; Received '{2:s}'".format(expected_output,
            ','.join(["UserSpeechFragment('"+f.role+"', '"+f.fragment_text+"')" for f in self.fragment_collection]),
             received_output)
    def test_dialogue_collection_to_string_agent(self):
        expected_output = 'hello how is it going?'
        received_output = '[NOT A STRING]'
        if self.is_string(get_agent_transcript(self.fragment_collection)):
            received_output = get_agent_transcript(self.fragment_collection)
        assert expected_output == get_agent_transcript(self.fragment_collection),\
           "Expected '{0:s}' from [{1:s}]; Received '{2:s}'".format(expected_output,
            ','.join(["UserSpeechFragment('"+f.role+"', '"+f.fragment_text+"')" for f in self.fragment_collection]),
             received_output)
    def test_dialogue_collection_to_string_caller(self):
        expected_output = 'hi great!'
        received_output = '[NOT A STRING]'
        if self.is_string(get_caller_transcript(self.fragment_collection)):
            received_output = get_caller_transcript(self.fragment_collection)
        assert expected_output == get_caller_transcript(self.fragment_collection),\
           "Expected '{0:s}' from [{1:s}]; Received '{2:s}'".format(expected_output,
            ','.join(["UserSpeechFragment('"+f.role+"', '"+f.fragment_text+"')" for f in self.fragment_collection]),
             received_output)
        


# a print_debugging_function
def print_out_filtration_results():
    ''' take a small sample, the first few records in the transcript file
        perform the row-ification and dialogue collection transformation
        then print each fragment in each dialogue
        
        run with:
             python -c 'import run_project;run_project.print_out_filtration_results()'
    '''
    lines = get_lines_from_transcript_file()
    lines = itertools.islice(lines, 10**3)
    for line in lines:
        row = line_to_feature_dictionary(line)
        dialogue_collection = transcript_to_dialogue_collection(row['transcript'])
        print '\t', dialogue_collection_to_string(dialogue_collection)
        print '\t', get_agent_transcript(dialogue_collection)
        print '\t', get_caller_transcript(dialogue_collection)
        print



def create_stopwords_set():
    ''' pulls stopwords from NLTK, a given name dataset and a last name dataset '''
    def get_words_from_file(file_path):
        ''' assuming 1 word per line, this function retrieves
            a list of words from a a file in utf-8 encoding '''
        with open(file_path) as fin:
            return [unicode(word.lower().strip(), 'utf-8') for word in fin]
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    firstnames = get_words_from_file(settings.GIVEN_NAMES_FILE_PATH)
    lastnames = get_words_from_file(settings.FAMILY_NAMES_FILE_PATH)
    return set(nltk_stopwords + firstnames + lastnames)

class StopWords(object):
    ''' don't worry about how this works for now.
        if you must know, stopwords is a static/class variable
        that is lazy instantiated. Retrieve the stopwords by calling
        StopWords.get_stopwords() '''
    stopwords = None
    @classmethod
    def get_stopwords(cls):
        if not cls.stopwords:
            cls.stopwords = create_stopwords_set()
        return cls.stopwords

def remove_stopwords_from_token_collection(tokens):
    '''
    If given a list of tokens, filter out any that are stopwords.
    Generally you can think of a token as an individual word, although
    if you get at all deeper into Natural Language Processing you'll
    find that usually the tokens are actually pieces of words with things
    like suffixes stripped off.

    Arguments:
    tokens: A list of individual tokens (as strings) to have stopwords
        filtered out.

    Returns:
    filtered_tokens: A list of the input tokens, minus the stopwords.

    Hint: You'll need to first get the list of stopwords, which you can do
        by calling StopWords.get_stopwords() and assigning the output of
        that to a variable. Then for each token in your list you'll need to
        check and see if it's in that list of stopwords.
        
    Test with:
        nosetests -v run_project:test_remove_stopwords_from_token_collection
    '''
    

def test_remove_stopwords_from_token_collection():
    input_tokens = 'devin is the coolest python programmer on the block'.split(' ')
    expected_output = ['coolest', 'python', 'programmer']
    assert expected_output == remove_stopwords_from_token_collection(input_tokens)


def remove_non_alphabetic_characters(dirty_string):
    '''
    Parse through a string and remove any characters that
    are not alphabetic (i.e. not a-z).

    Arguments:
    dirty_string: A string which may (or may not!) have non-alphabetic
        characters.

    Returns:
    clean_string: A version of dirty_string with all non-alphabetic
        characters removed. If all of dirty_string's characters were
        non-alphabetic, return an empty string ('').

    Hint: You can parse a string character-by-character, just as you would a
        list or array. You can also append characters to a string with the
        plus sign (+).

    
    Test with:
        nosetests -v run_project:test_remove_non_alphabetic_characters
    '''
    
def test_remove_non_alphabetic_characters():
    input_string = 'abc 123 easy as hello, world 1 ! \xef\xbb\xbf \t \n'
    expected_output = 'abceasyashelloworld'
    assert expected_output == remove_non_alphabetic_characters(input_string)


def remove_short_tokens_from_collection(tokens, min_length=4):
    '''
    Go through a list of tokens and filter out any short ones.

    Arguments:
    tokens: A list of strings, where each string is a token.
    min_length (4): A keyword argument allowing the minimum
        length to filter on to be changed.
    
    Returns:
    filtered_tokens: A list of all the strings in 'tokens' with
        >= min_length characters.

    Hint: If you're clever, you can do this in as little as one line :).
    
    Test with:
        nosetests -v run_project:test_remove_short_tokens_from_collection
    '''

def test_remove_short_tokens_from_collection():
    input_tokens = 'andrew is the coolest python programmer on the block'.split(' ')
    expected_output = ['andrew', 'coolest', 'python', 'programmer']
    assert expected_output == remove_short_tokens_from_collection(input_tokens, min_length=6)


def tokenize_and_clean_transcript(transcript, min_length=4):
    '''
    This function acts as a wrapper to several functions:
    1. Because we want to treat "capital one" as a single thing,
        but the pipeline splits everything up into individual words,
        we replace any instances of "capital one" with a single word.
    2. We then use the nltk function 'word_tokenize' to split the transcript
        into a list of individual words (the tokens).
    3. We remove any punctuation, numbers, and any other oddness by applying
        the 'remove_non_alphabetic_characters' function to each token. For more
        details about the 'map' function used here, see:
        https://docs.python.org/2/howto/functional.html#built-in-functions.
    4. Remove any stopwords from the list of tokens with the
        'remove_stopwords_from_token_collection' function.
    5. Finally, remove any remaining short tokens with the
        'remove_short_tokens_from_collection_function'.

    Test, just to make sure all the pieces are working, with:
        nosetests -v run_project:test_tokenize_and_clean_transcript

    '''
    # custom entity parsing
    transcript = transcript.replace('capital one', 'capitalone')

    # tokenize
    transcript_tokenized = nltk.word_tokenize(transcript)

    # clean up punctuation and remove empty tokens
    no_punk = map(remove_non_alphabetic_characters, transcript_tokenized)

    #Remove stopwords:
    no_stopwords = remove_stopwords_from_token_collection(no_punk)
    
    # remove short words and empty tokens
    no_shorts = remove_short_tokens_from_collection(no_stopwords, min_length=min_length)
    return no_shorts

def test_tokenize_and_clean_transcript():
    input_string = 'Macy is the coolest, most fabulous pooch on the playground'.lower()
    expected_output = ['coolest', 'fabulous', 'playground']
    assert expected_output == tokenize_and_clean_transcript(input_string, min_length=3)


# a print_debugging_function
def print_out_cleaning_results():
    ''' sanity checking for the text processing pipeline

        run with:
             python -c 'import run_project;run_project.print_out_cleaning_results()' > ~/Desktop/results.txt
    '''

    lines = get_lines_from_transcript_file()
    lines = itertools.islice(lines, 10**2)
    for line in lines:
        row = line_to_feature_dictionary(line)
        dialogue_collection = transcript_to_dialogue_collection(row['transcript'])

        transcript = dialogue_collection_to_string(dialogue_collection)
        vectorizable_tokens = tokenize_and_clean_transcript(transcript)

        print transcript
        print tokens_clean
        print vectorizable_tokens
        print


# a print_debugging_function
def print_out_cleaning_results():
    ''' run with:
             python -c 'import run_project;run_project.print_out_cleaning_results()' > ~/Desktop/results.txt
    '''

    lines = get_lines_from_transcript_file(INPUT_FILE_PATH)
    lines = itertools.islice(lines, 10**3)
    for line in lines:
        row = line_to_feature_dictionary(line)
        dialogue_collection = transcript_to_dialogue_collection(row['transcript'])

        transcript = dialogue_collection_to_string(dialogue_collection)
        vectorizable_tokens = tokenize_and_clean_transcript(transcript)

        print dialogue_collection_pretty_print(dialogue_collection)
        print 



def get_transcripts_collection():
    '''
    Here's the payout for all your hard work - the one-stop-shop
    text cleaning function! As a reward for your efforts we've
    gone ahead and filled it in for you :). Here's what it does:

    1. Load the transcript file.
    2. Iterate through the file:
        2a. Parse the line into a dictionary of features.
        2b. Convert the transcript part of the line into a collection of
            UserSpeechFragment tuples for each uninterruped string of words
            by a single speaker.
        2c. Merge the list of UserSpeechFragments back into a single string.
        2d. Convert that string into a list of individual tokens (words), remove
            punctuation, strip out stopwords, and delete small words.
        2e. Rejoin the cleaned tokens back into a single string, add to a list.

    Note that the function returns both the fully processed transcripts as well
    as the raw ones - this will be valuable later, when we want to check the results
    of our modeling (the processed transcripts are even more unintelligible than
    the originals, if you can believe it). 
    '''
    transcript_collection = []
    raw_transcript_collection = []

    lines = get_lines_from_transcript_file(INPUT_FILE_PATH)
    for line in lines:
        row = line_to_feature_dictionary(line)
        dialogue_collection = transcript_to_dialogue_collection(row['transcript'])
        raw_transcript_collection.append(row['transcript'])

        transcript = dialogue_collection_to_string(dialogue_collection)
        vectorizable_tokens = tokenize_and_clean_transcript(transcript)

        transcript_collection.append(' '.join(vectorizable_tokens))

    return raw_transcript_collection, transcript_collection

def vectorize_transcript_collection(transcript_collection):
    '''
    To convert our transcripts into features, we use a process
    called vectorization, which is the standard way to convert
    raw text into useful features for machine learning algorithms
    (see http://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation
    for some more info about vectorization and its application in
    text analysis).

    Specifically we'll be using the TfidfVectorizor, which will weight
    the transcripts by how uncommon the words they use are - the less a word
    appears the more likely it is to be important for classifying/understanding
    a transcript. More information on tf-idf weighting can be found at the
    above link as well as at http://en.wikipedia.org/wiki/Tf%E2%80%93idf).

    Initialize the TfidfVectorizer with the following parameters:
        sublinear_tf=True
        max_df=0.5
        max_features=100
        stop_words='english'

    Then train the vectorizer on the transcript_collection via the '.fit' method
    of the vectorizer. For details about the TfidfVectorizer, see the documentation:
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    Arguments:
    transcript_collection: The list of call transcript strings you want
        to vectorize.

    Returns:
    vectorizer: The output of the fit operation
        on the transcript_collection.

    Hint: As you'll see at the top of this file,
        the TfidfVectorizer has already been imported
        for you. From this point initializing an instance of the
        vectorizer and running 'fit' should be one line
        of code each.
        
    Test with:
        nosetests -v run_project:test_vectorize_transcript_collection
    '''

class test_vectorize_transcript_collection(object):
    @classmethod
    def setUpClass(self):
        input_collection = ['hello my name is andrew',
                    'hello my name is devin',
                    'hi my name is chris',
                    'hi im lia']
        self.vectorizer = vectorize_transcript_collection(input_collection)
        self.correct_params = {'analyzer': u'word',
             'binary': False,
             'charset': None,
             'charset_error': None,
             'decode_error': u'strict',
             'dtype': np.int64,
             'encoding': u'utf-8',
             'input': u'content',
             'lowercase': True,
             'max_df': 0.5,
             'max_features': 100,
             'min_df': 1,
             'ngram_range': (1, 1),
             'norm': u'l2',
             'preprocessor': None,
             'smooth_idf': True,
             'stop_words': 'english',
             'strip_accents': None,
             'sublinear_tf': True,
             'token_pattern': u'(?u)\\b\\w\\w+\\b',
             'tokenizer': None,
             'use_idf': True,
             'vocabulary': None}
        self.output_params = {'analyzer': u'word',
             'binary': False,
             'charset': None,
             'charset_error': None,
             'decode_error': u'strict',
             'dtype': np.int64,
             'encoding': u'utf-8',
             'input': u'content',
             'lowercase': True,
             'max_df': 0.5,
             'max_features': 100,
             'min_df': 1,
             'ngram_range': (1, 1),
             'norm': u'l2',
             'preprocessor': None,
             'smooth_idf': True,
             'stop_words': 'english',
             'strip_accents': None,
             'sublinear_tf': True,
             'token_pattern': u'(?u)\\b\\w\\w+\\b',
             'tokenizer': None,
             'use_idf': True,
             'vocabulary': None}
        if hasattr(self.vectorizer,'get_params'):
            self.output_params = self.vectorizer.get_params()

    def test_vectorizer_fits(self):
        assert hasattr(self.vectorizer,'vocabulary_') == True, 'vectorize_transcript_collection does not return '\
            'a model that has been fit to data.'
    def test_vectorizer_params_exist(self):
        assert hasattr(self.vectorizer,'get_params') == True, "vectorize_transcript_collection doesn't "\
          'initialize the TfidfVectorizer.'
    def test_vectorizer_params(self):
        for key in self.correct_params.keys():
            yield self.check_correct_vectorizer_params, key
    def check_correct_vectorizer_params(self, key):
        assert self.correct_params[key] == self.output_params[key],'Incorrect value for '\
          '{0} parameter. Expected {1}. Received {2}.'.format(key, self.correct_params[key],
                                                              self.output_params[key])

def run_k_means(text_features):
    '''
    Initialize and fit a K-Means clustering algorithm to the
    vectorized transcripts. This will take two steps:
    1. Initialize the K-Means algorithm. We'll use the
        scikit-learn MiniBatchKMeans algorithm (imported
        at the top of this file) rather than a regular
        K-Means algorithm because it is faster without
        sacrificing much accuracy (for a description and comparison,
        see http://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans).
        Use the following parameters in the initialization:
            init='k-means++'
            n_clusters=10
            n_init=10
            init_size=1000
            batch_size=1000
            verbose=1
    2. Use the .fit function of your initialized MiniBatchKMeans to cluster the data.

    Arguments:
    text_features: The vectorized transcripts, from 'vectorize_transcript_collection'

    Returns:
    fit_kmeans: The MiniBatchKMeans object, initialized with the parameters above and
        fit to the text_features.

    Hint: Scikit-Learn uses classes extensively to provide persistence in its models.
        This is really useful, since not only do you want to train your model but you
        of course want to test it out, and then ultimately may end up running it over
        and over on different data. It does make things a little complicated to learn,
        however. If you're having difficulty, take a look at the tutorial
        (http://scikit-learn.org/stable/tutorial/basic/tutorial.html) and the examples
        (http://scikit-learn.org/stable/auto_examples/index.html).
        
    Test with:
        nosetests -v run_project:test_run_k_means
        
    '''

class test_run_k_means(object):
    @classmethod
    def setUpClass(self):
        data = np.array([0.78528828, 0.6191303, 0.78528828, 0.6191303 , 0.78528828,
                        0.6191303, 0.61761437, 0.61761437, 0.48693426, 0.4869, 0.4869, 0.4869,
                        0.25, 0.34, 0.5])#15
        row = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 8, 9, 10])
        col = np.array([0, 3, 2, 3, 1, 4, 6, 5, 4, 1, 4, 0, 6, 3, 4])
        from scipy.sparse import csr_matrix
        self.vectorized_data = csr_matrix((data,(row,col)))
        self.fit_kmeans = run_k_means(self.vectorized_data)
        self.correct_params = {'batch_size':1000,
                  'compute_labels':True,
                  'init':'k-means++',
                  'init_size':1000,
                  'max_iter':100,
                  'max_no_improvement':10,
                  'n_clusters':10,
                  'n_init':10,
                  'random_state':None,
                  'reassignment_ratio':0.01,
                  'tol':0.0,
                  'verbose':1
                  }
        self.output_params = {'batch_size':1000,
                  'compute_labels':True,
                  'init':'k-means++',
                  'init_size':1000,
                  'max_iter':100,
                  'max_no_improvement':10,
                  'n_clusters':10,
                  'n_init':10,
                  'random_state':None,
                  'reassignment_ratio':0.01,
                  'tol':0.0,
                  'verbose':1
                  }
        if hasattr(self.fit_kmeans,'get_params'):
            self.output_params = self.fit_kmeans.get_params()

    def test_kmeans_fits(self):
        assert hasattr(self.fit_kmeans,'labels_') == True, 'run_k_means does not return '\
            'a model that has been fit to data.'
    def test_kmeans_params_exist(self):
        assert hasattr(self.fit_kmeans,'get_params') == True, "run_k_means doesn't "\
          'initialize the MiniBatchKMeans model.'
    def test_kmeans_params(self):
        for key in self.correct_params.keys():
            yield self.check_correct_kmeans_params, key
    def check_correct_kmeans_params(self, key):
        assert self.correct_params[key] == self.output_params[key],'Incorrect value for '\
          '{0} parameter. Expected {1}. Received {2}.'.format(key, self.correct_params[key],
                                                              self.output_params[key])

def split_data_into_samples(transcript_collection, cluster_labels, test_frac=0.2):
    '''
    We don't want to train a predictive model on our entire labeled
    sample - if we do that we won't be able to accurately gauge
    its performance. So this function splits the data and labels
    into two groups: the training set (the data the model will be
    fit on) and the test set (the data we'll use to estimate how
    good the model is). The important thing is to make this split happen
    RANDOMLY in the data - if your data has any kind of ordering to it,
    you'll bias your model if you do any kind of non-random sampling.

    Arguments:
    transcript_collection: A list of strings, one for each transcript.
    cluster_labels: A list of labels for each of those transcripts, from k_means.labels_
    test_frac (0.2): What fraction of the data should be used for testing
        (20% is a good default).

    Returns:
    train_transcripts, test_transcripts, train_labels, test_labels: A tuple
        of lists, giving (in order) the training transcripts, the testing transcripts,
        the training labels, and the testing labels.

    Hint: You can do this in a few different ways, but scikit-learn has
        a function called 'train_test_split' (imported at the top of this file)
        that's designed to do exactly this.

    Test with:
        nosetests -v run_project:test_split_data_into_samples
    
    '''

class test_split_data_into_samples(object):
    def setup(self):
        self.transcripts = ['a','b','c','d','e','f','g','h','i','j']
        self.labels = [0,0,1,0,0,1,2,1,2,0]
    def test_split_data_into_samples_output_format(self):
        output = split_data_into_samples(self.transcripts,self.labels)
        assert hasattr(output,'__iter__') and len(output) == 4, "output format of"\
              " split_data_into_samples is wrong, needs to be a 4-element tuple"
    def test_split_data_into_samples_random(self):
        train_t1, test_t1, train_l1, test_l1 = split_data_into_samples(self.transcripts,self.labels)
        train_t2, test_t2, train_l2, test_l2 = split_data_into_samples(self.transcripts,self.labels)
        assert np.array_equal(train_t1,train_t2) == False\
          and np.array_equal(train_l1,train_l2) == False,\
          "The data are not being sufficiently randomized."
    def test_split_data_into_samples_same_size(self):
        train_t1, test_t1, train_l1, test_l1 = split_data_into_samples(self.transcripts,self.labels)
        train_t2, test_t2, train_l2, test_l2 = split_data_into_samples(self.transcripts,self.labels)
        assert len(train_t1) == len(train_t2)\
          and len(train_l1) == len(train_l2),\
          "The data are not being split into consistent training and testing subsamples."
    def test_split_data_into_samples_proper_split(self):
        train_t1, test_t1, train_l1, test_l1 = split_data_into_samples(self.transcripts,self.labels,
                                                                       test_frac=0.5)
        assert len(train_t1) == len(test_t1),\
          "The data are not being split appropriately for test_frac=0.5."

def train_logistic_regression(vectorized_training_transcripts,training_labels):
    '''
    Initialize and train a multiclass logistic regression classifier on the
    training data, using scikit-learn's LogisticRegression class with default
    parameters.

    Arguments:
    vectorized_training_transcripts: The fully processed and vectorized call
        center transcripts that make up the training set.
    training_labels: The k-means clustering labels assigned to the training
        set transcripts.

    Returns:
    fit_logistic: An instance of scikit-learn's LogisticRegression class,
        initialized with default parameters and fit to the training data.

    Hint: You can look up documentation and examples for the LogisticRegression
        class by Googling 'scikit-learn logistic regression'. You'll notice,
        however, that using it is virtually identical to how we used
        scikit-learn's k-means clustering. This type of scheme is applied
        consistently across scikit-learn's entire suite of predictive
        models, which makes it really easy to swap between models when
        you're doing exploratory analysis.

    Test with:
        nosetests -v run_project:test_train_logistic_regression
    '''
    
class test_train_logistic_regression(object):
    @classmethod
    def setup(self):
        self.transcripts = np.random.rand(20,20)
        self.labels = np.random.randint(0,2,20)
        self.fit_logreg = train_logistic_regression(self.transcripts,self.labels)
        #print dir(self.fit_logreg)
        self.correct_params = {'C': 1.0,
             'class_weight': None,
             'dual': False,
             'fit_intercept': True,
             'intercept_scaling': 1,
             'penalty': 'l2',
             'random_state': None,
             'tol': 0.0001}
        self.output_params = {'C': 1.0,
             'class_weight': None,
             'dual': False,
             'fit_intercept': True,
             'intercept_scaling': 1,
             'penalty': 'l2',
             'random_state': None,
             'tol': 0.0001}
    def test_train_logistic_regression_fits(self):
        assert hasattr(self.fit_logreg,'coef_') == True, 'train_logistic_regression '\
          'does not return a model that has been fit to data.'
    def test_train_logistic_regression_params_exist(self):
        assert hasattr(self.fit_logreg,'get_params') == True, "train_logistic_regression doesn't "\
          'initialize the LogisticRegression model.'
    def test_train_logistic_regression_params(self):
        for key in self.correct_params.keys():
            yield self.check_correct_logreg_params, key
    def check_correct_logreg_params(self, key):
        assert self.correct_params[key] == self.output_params[key],'Incorrect value for '\
          '{0} parameter. Expected {1}. Received {2}.'.format(key, self.correct_params[key],
                                                              self.output_params[key])

    
def print_cluster_examples(transcripts,labels,num_examples=10,file_extension='.txt'):
    '''
    Print a small number of examples of transcripts from each cluster to files,
    for by-eye inspection.

    Arguments:
    transcripts: A list of strings, one for each transcript.
    labels: The labels assigned to the transcripts by the clustering algorithm.
    num_examples (10): The number of transcripts to write for each cluster.
    file_extension (.txt): The extension to append to the file. 
    '''
    unique_labels = np.unique(labels)
    np_transcripts = np.array(transcripts)
    for i,label in enumerate(unique_labels):
        transcripts_with_label = np_transcripts[labels == label]
        output_filename = 'cluster_{0}_examples{1:s}'.format(label,file_extension)
        np.savetxt(output_filename,transcripts_with_label[:num_examples],fmt='%s')
        
def vectorize_and_model(raw_transcripts, transcript_collection):
    ''' run with:
             python -c 'import run_project;run_project.vectorize_and_model()'
    '''

    text_vectorizer = vectorize_transcript_collection(transcript_collection)
    text_features = text_vectorizer.transform(transcript_collection)

    # matrix logging
    logging.info('X - n_samples: {0}, n_features: {1}'.format(*text_features.shape))

    #Use clustering to make labels for the transcripts
    k_means = run_k_means(text_features)
    k_means_labels = k_means.labels_

    #print cluster information to the log and save sample transcripts
    logging.info('clusters and cluster frequency: {}'.format(collections.Counter(k_means_labels)))
    print_cluster_examples(raw_transcripts, k_means_labels, num_examples=5, file_extension='_temp.nogit')

    # we started with unlabeled data, and now we've labeled it using an
    # unsupervised machine learning algorithm. Let's see if we can reconstruct 
    # those classifications with a supervised algorithm.
    # but we don't want to train and test on the same data, so let's construct
    # a supervised modeling pipeline with a train/test split:
    #   http://scikit-learn.org/stable/modules/pipeline.html
    #   http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#building-a-pipeline
    #   http://stats.stackexchange.com/questions/95797/how-to-split-the-dataset-for-cross-validation-learning-curve-and-final-evaluat

    #Split the data into training and test sets:
    transcript_train, transcript_test, label_train, label_test = split_data_into_samples(transcript_collection,k_means_labels)

    #Vectorize the train and test data:
    vectorized_transcript_train = text_vectorizer.transform(transcript_train)
    vectorized_transcript_test = text_vectorizer.transform(transcript_test)

    #Initialize the Logistic Regression:
    logreg = LogisticRegression()
    logreg.fit(vectorized_transcript_train,label_train)
    predicted = logreg.predict(vectorized_transcript_test)
    accuracy = np.mean(predicted == label_test)
    print "Done: accuracy = {0:.2f} on {1:d}"\
      " testing transcripts ({2:d} transcripts used for training)".format(accuracy,len(label_test),len(label_train))
    
    # text_clf_trained = text_clf_pipeline.fit(X_train, y_train)
    # predicted = text_clf_trained.predict(X_test)
    
    # y_test_category_match_ratio = np.mean(predicted == y_test)
    # logging.info('all_data_category_match_ratio: {}'.format(y_test_category_match_ratio))


    # CONGRATULATIONS!
    # You just built a real natural language processing pipeline
    # that can automatically categorize new call center transcripts into auto-discovered topic
    # buckets. Pretty gosh-darn cool if you ask me.

    # Where do you go from here?
    #   interesting question... try modifying the transcript fetching function to
    #   pull only agent or caller transcripts and re-running the pipeline
    #
    # There are lots of perspectives you could take at this point:
    #       business would want to to understand the clustered data and what the clusters mean
    #       statisticians would want to validate the model and try different clustering and classification algorithms
    #       developers would want to know how the algorithms scale and if stream processing is possible
    #       MRO would want validation and monitoring
    #
    # If you don't see your use case or you have strong feelings about a particular topic,
    # let me know and I'll do my best to implement your request - let me know here:
    #       https://github.kdc.capitalone.com/BigDataAcademy/Small_Data/issues?state=open
    #


def main():
    '''  '''
    print "Starting"
    raw_transcripts, transcript_collection = get_transcripts_collection()
    print "Finished Collecting Transcripts"
    vectorize_and_model(raw_transcripts, transcript_collection)
    

if __name__ == '__main__':
    main()

