#===========PARALELL CHAINING=================
model1 = ChatGroq(model="llama-3.3-70b-versatile")
model2 = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template="Generate a short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = "generate a 5 short question answers from the following text \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template= "Merge the provided notes and quizz into a single document \n notes ->{notes} and {quiz}",
    input_variables = ["notes","quiz"]
)
parser= StrOutputParser()

parallel_chain = RunnableParallel(
    notes = prompt1|model1,
    quiz = prompt2|model2
)
final_chain = parallel_chain|prompt3|model1|parser


response = final_chain.invoke({"text":"""A Transformer is a modern architecture used in programming for building complex AI models. In the context of your Programming Paradigms syllabus and provided documents, here are the detailed notes:

1. Conceptual Foundation

Abstraction of Memory: Like all variables in imperative languages , a Transformer is an abstraction that manages memory cells.

Von Neumann Architecture: The Transformer operates within the standard architecture consisting of Memory (storing data and instructions) and a Processor (modifying memory contents).
+1

Imperative Nature: Current implementations (like your PyTorch code) follow the Imperative Paradigm, where the goal is to understand and change the machine state through a sequence of statements.

2. Core Components & Binding (Unit II)

Word Embeddings: These are vectors, which are data structures composed of a fixed number of components of the same type. They represent an Entity-Attribute binding between a token and its numerical vector.
+1

Positional Encoding: This process adds temporal information to the word embeddings. In programming terms, it is a mathematical operation that ensures the "machine state" accounts for the order of data.

Attention Mechanism:

This is the "brain" of the Transformer.

It involves Arithmetic Expressions and Relational/Boolean Expressions to calculate how much "attention" one token should pay to another.

It uses overloaded operators (like matmul or + in your code) to handle matrix computations."""
})
print(response)
print(final_chain.get_graph().print_ascii())

""" +---------------------------+          
          | Parallel<notes,quiz>Input |          
          +---------------------------+          
                ***           ***                
              **                 **              
            **                     **            
+----------------+            +----------------+ 
| PromptTemplate |            | PromptTemplate | 
+----------------+            +----------------+ 
         *                             *         
         *                             *         
         *                             *         
   +----------+                  +----------+    
   | ChatGroq |                  | ChatGroq |    
   +----------+*                 +----------+    
                ***           ***                
                   **       **                   
                     **   **                     
         +----------------------------+          
         | Parallel<notes,quiz>Output |          
         +----------------------------+          
                        *                        
                        *                        
                        *                        
               +----------------+                
               | PromptTemplate |                
               +----------------+                
                        *                        
                        *                        
                        *                        
                  +----------+                   
                  | ChatGroq |                   
                  +----------+                   
                        *                        
                        *                        
                        *                        
              +-----------------+                
              | StrOutputParser |                
              +-----------------+                
                        *                        
                        *                        
                        *                        
            +-----------------------+            
            | StrOutputParserOutput |            
            +-----------------------+            
                        *                        
                        *                        
                        *                        """