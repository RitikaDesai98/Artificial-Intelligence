# import StringDouble
# import ExtractGraph
from pycorenlp import StanfordCoreNLP

class ExtractOpinions:
    # Extracted opinions and corresponding review id is saved in extracted_pairs, where KEY is the opinion and VALUE
    # is the set of review_ids where the opinion is extracted from.
    # Opinion should in form of "attribute, assessment", such as "service, good".
    extracted_opinions = {}
    
    def __init__(self):
        return

    def extract_pairs(self, review_id, review_content):
        nlp = StanfordCoreNLP('http://localhost:9000')
        props = {
        'annotators': ' tokenize, ssplit, lemma, depparse, sentiment, ner, pos',
        'outputFormat': 'json',
        'openie.triple.strict':'true',
        'timeout': 50000,
        }
        output = nlp.annotate(review_content,properties=props)
        parts_of_speech = {}
        for sentence in output['sentences']:            
            for term in sentence["tokens"]:
                parts_of_speech[term["word"] ]= term["pos"]
            dependencies = [sentence["enhancedDependencies"] for item in output]
            for i in dependencies:
                opinions = ""
                for enhancedDependencies in i:                                   
                    if enhancedDependencies['dep'] == 'nsubj' and parts_of_speech[enhancedDependencies['governorGloss']] == 'JJ' and parts_of_speech[enhancedDependencies['dependentGloss']] == 'NN':
                        opinions = enhancedDependencies['dependentGloss'].lower() + ", " + enhancedDependencies['governorGloss'].lower()
                    if enhancedDependencies['dep'] == 'amod':
                        opinions = enhancedDependencies['governorGloss'].lower() + ", " + enhancedDependencies['dependentGloss'].lower()         
                if opinions != "":
                    if opinions not in self.extracted_opinions.keys():
                        self.extracted_opinions[opinions] = [review_id]
                    else:
                        ids = self.extracted_opinions[opinions]
                        if review_id not in ids:
                            ids.append(review_id)
                            self.extracted_opinions[opinions] = ids
                
        return(self.extracted_opinions)
        
#         self.extracted_opinions = {'service, good': [1, 2, 5], 'service, excellent': [4, 6]}
