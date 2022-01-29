import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u'Tesla is looking to set up a production unit in India in 2021 of Model S and Model Y . The Model-S and Model-Y are the bestselling models of Tesla till date.')

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text + '-' + ent.label_ + '-' + str(spacy.explain(ent.label_)))
    else:
        print("No Entity found")
    
show_ents(doc)

from spacy.tokens import Span
ORG = doc.vocab.strings[u'ORG']
new_ent = Span(doc, 0, 1, label = ORG)
doc.ents = list(doc.ents) + [new_ent]

show_ents(doc)

# to import multiple similar entities at the same time 

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)
#Creating the desired phrase list 
phrase_list = ['Model S','Model Y','Model-S','Model-Y']
phrase_pattern = [nlp(text) for text in phrase_list]
# Adding it to the matcher
matcher.add('newproduct',None,*phrase_pattern)

matches = matcher(doc)
# Creating Spans for each span and then create Named entities from them .
from spacy.tokens import Span

PROD = doc.vocab.strings[u'PRODUCT']

new_ents = [Span(doc, match[1],match[2],label=PROD) for match in matches]

doc.ents = list(doc.ents) + new_ents

show_ents(doc)