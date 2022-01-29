import spacy 
nlp = spacy.load('en_core_web_sm')

from spacy import displacy 

with open('peterrabbit.txt') as f:
    doc = nlp(f.read())
    
sents = [sent for sent in doc.sents]

for token in sents[4]:
    print(f'{token.text:{8}} {token.pos_:{8}} {token.tag_:{8}} {spacy.explain(token.tag_)}')

POS_counts = doc.count_by(spacy.attrs.POS)

for k,v in sorted(POS_counts.items()):
    print(f'{k} {doc.vocab[k].text:{5}}:{v}')

total_pos = 0
noun_count = 0
for k,v in sorted(POS_counts.items()):
    total_pos = total_pos + v
    if k == 91:
        noun_count = noun_count + v
  
doc1 = nlp(str(sents[2]))
    
i = 0             
doc2 = nlp(str(sents[0]))               
for ent in doc2.ents[:2]:
        print(ent.text + '-' + ent.label_ + '-' + str(spacy.explain(ent.label_)))

        
ners = [doc for doc in sents if doc.ents]




    