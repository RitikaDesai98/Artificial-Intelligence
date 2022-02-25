class ExtractGraph:

    # key is head word; value stores next word and corresponding probability.
    graph = {}

    sentences_add = "assign1_sentences.txt"

    def __init__(self):
        head_word = '<s>'
        self.graph['<s>'] = {}
        number = 0
        # Extract the directed weighted graph, and save to {head_word, {tail_word, probability}}
        with open(self.sentences_add) as f:
            #Read each sentence the text file
            for sentence in f:              
                #Read each word in the sentence  
                for current_word in sentence.split():
                    #Check if the current_word is not <s> and head_word is not </s>
                    if current_word != '<s>' and head_word != '</s>':
                        #Check for head_word in the graph keys
                        if head_word not in self.graph.keys():
                            #If not present, add it
                            self.graph[head_word] = {}
                        #Check if current word is in graph with key as head_word
                        if current_word not in self.graph[head_word].keys():
                            #If not present, add it as {head_word, {current_word, occurrences}}
                            self.graph[head_word][current_word] = 1
                        else:
                            #If it is present, add one to the occurrences
                            self.graph[head_word][current_word] += 1
                    #Now set the head_word as the current_word
                    head_word = current_word

        #Use a for loop to iterate over all the keys of graph
        for i in self.graph.keys():
            #Use a second for loop to iterate over the graph's keys that have value i
            for j in self.graph[i].keys():
                #Add number of times it appears
                number += self.graph[i][j]
            #Again iterate over the graph's keys that have value i
            for j in self.graph[i].keys():
                #Now calculate the probability of that pair
                self.graph[i][j] = self.graph[i][j]/number                
            number = 0
        return

    def getProb(self, head_word, tail_word):
        #if head_word or tail_word isn't present in the graph, return 0
        if head_word not in self.graph.keys() or tail_word not in self.graph[head_word].keys():
            return 0
        #else return graph with values [head_word][tail_word] to get the probability of the pair
        return self.graph[head_word][tail_word]