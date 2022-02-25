import math
import StringDouble
import heapq

class BeamSearch:

    graph = []

    def __init__(self, input_graph):
        self.graph = input_graph
        return

    def beamSearchV1(self, pre_words, beamK, maxToken):
    # Just call beamSearchV2 with lambda = 0 as it implements the same thing as a regular beam search
        return self.beamSearchV2(pre_words, beamK, 0, maxToken)


    def beamSearchV2(self, pre_words, beamK, param_lambda, maxToken):
        # Initialize the values
        wordGraph = self.graph.graph
        sentence = pre_words
        probability = 0.0
        pHeap = []
        pWord = "<s>"
        #Use a for loop to iterate over all the words present in the previous words by splitting the words
        for word in pre_words.split():
            if word != "<s>":
                #Use the score formula to calculate the log probability
                probability = probability + math.log(wordGraph[pWord][word])
                #Set the next word as previous word
                pWord = word
        self.heap_push(pHeap, beamK, probability, False, sentence)
        # while complete is true
        while True:
            cHeap = []
            #Use a for loop to iterate over the values in pheap
            for (probability, complete, sentence) in pHeap:
                #if complete is true, then push to heap
                if complete == True:
                    self.heap_push(cHeap, beamK, probability, True, sentence)
                #if not, then set head_word as last word
                else:
                    head_word = sentence.split()[-1]
                    #use a if loop to iterate over the keys of graph with value i for the tail_word
                    for tail_word in self.graph.graph[head_word].keys():
                        #if tail_word is </s> means sentence is complete
                        if tail_word == "</s>":
                            length_norm = (len(sentence.split()) + 1) ** param_lambda
                            self.heap_push(cHeap, beamK, probability + math.log(wordGraph[head_word][tail_word])/length_norm, True, sentence + " " + tail_word)
                        #if not, then continue looping
                        else:
                            self.heap_push(cHeap, beamK, probability + math.log(wordGraph[head_word][tail_word]), False, sentence + " " + tail_word)
            #get the maximum score from the current heap
            (probability, complete, sentence) = max(cHeap)
            #Once the sentence is complete, return the probability
            if complete == True and len(sentence.split()) <= maxToken:
                return StringDouble.StringDouble(sentence, math.exp(probability))
            #Set the previous heap as the current heap
            pHeap = cHeap
    
    def heap_push(self, heap_q, beamK, sentence_probability, complete, sentence):
        heapq.heappush(heap_q, (sentence_probability, complete, sentence))
        if len(heap_q) > beamK:
            heapq.heappop(heap_q)