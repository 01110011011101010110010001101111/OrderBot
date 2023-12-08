import spacy

class OrderToFoods:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    def extract_food_items(self, text):
        doc = self.nlp(text)

        # Define a list of relevant POS (Part-of-Speech) tags for food items
        food_pos_tags = ["NOUN", "PROPN"]

        # Extract food items based on POS tags
        food_items = [token.text for token in doc if token.pos_ in food_pos_tags]

        return food_items

    def find_modifier(self, sentence):
        doc = self.nlp(sentence)

        neg_words = ["without", "no", "none", "hold"]

        for token in doc:
            if token.text.lower() in neg_words: 
                # Check if the word has a dependent (what it modifies)
                if token.dep_ == "prep" and token.head.dep_ == "ROOT":
                    # If it's a preposition modifying the root, find its dependent
                    modifier = [child.text for child in token.children]
                    return modifier
                elif token.dep_ == "advcl" and token.head.dep_ == "ROOT":
                    # If it's an adverbial clause modifying the root, find its dependent
                    modifier = [child.text for child in token.children]
                    return modifier
                elif token.dep_ == "neg":
                    # If it's a negation word, find its dependent
                    modifier = [child.text for child in token.children]
                    return modifier

        return None
    def execute(self, order):
        return set(self.extract_food_items(order)) - set(self.find_modifier(order))

# Example usage:
nlp = OrderToFoods()
sentence = "Can I have a sandwich without chicken?"
print(f"foods wanted: {nlp.execute(sentence)}")



# import spacy
# 
# def find_modification(sentence, preposition):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(sentence)
# 
#     modification_dict = {}
# 
#     # Iterate through tokens to find the preposition and its modifier
#     for token in doc:
#         if token.text.lower() == preposition.lower() and token.dep_ == "prep":
#             # Find the head of the preposition (the word it modifies)
#             head_token = token.head
#             modification_dict[token.text] = head_token.text
# 
#     return modification_dict
# 
# # Example usage:
# sentence ="I want a pizza without onions." #  "She left without saying goodbye."
# preposition = "without"
# modifications = find_modification(sentence, preposition)
# 
# print(f"The word modified by '{preposition}' is: {modifications.get(preposition, 'Not found')}")



# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# 
# def extract_wants_and_doesnt_want(sentence):
#     # Load BERT pre-trained model and tokenizer for sentiment analysis
#     model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
# 
#     # Sentiment analysis pipeline using BERT
#     classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# 
#     # Classify sentiment using BERT
#     result = classifier(sentence)[0]
# 
#     # Extract positive and negative sentiments
#     if result['label'] == 'POSITIVE':
#         wanted_items = [sentence]
#         unwanted_items = []
#     else:
#         wanted_items = []
#         unwanted_items = [sentence]
# 
#     return wanted_items, unwanted_items
# 
# # Example usage:
# order_sentence = "I want a large pepperoni pizza, but I don't want any onions."
# wanted, unwanted = extract_wants_and_doesnt_want(order_sentence)
# 
# print("Wanted items:", wanted)
# print("Unwanted items:", unwanted)



# import nltk
# from nltk import pos_tag, ne_chunk
# from nltk.tokenize import word_tokenize
# from nltk.sentiment import SentimentIntensityAnalyzer
# 
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# 
# nltk.download('punkt')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
# 
# def analyze_sentence(sentence):
#     print(sentence)
# 
#     # Tokenization
#     tokens = word_tokenize(sentence)
# 
#     # Part-of-Speech Tagging
#     pos_tags = pos_tag(tokens)
# 
#     # Named Entity Recognition
#     named_entities = ne_chunk(pos_tags)
# 
#     # Sentiment Analysis
#     sentiment_analyzer = SentimentIntensityAnalyzer()
#     sentiment_scores = sentiment_analyzer.polarity_scores(sentence)
#     print(sentiment_scores)
# 
#     # Identify positive and negative associations
#     positive_items = [token for token, pos in pos_tags if pos in ['JJ', 'NN'] and sentiment_scores[token] > 0]
#     negative_items = [token for token, pos in pos_tags if pos in ['JJ', 'NN'] and sentiment_scores[token] < 0]
# 
#     return {
#         'tokens': tokens,
#         'pos_tags': pos_tags,
#         'named_entities': named_entities,
#         'sentiment_scores': sentiment_scores,
#         'positive_items': positive_items,
#         'negative_items': negative_items
#     }
# 
# # Example sentence
# sentence = "I want a sandwich without chicken" # "I love my new smartphone, but the battery life is terrible."
# 
# analysis_result = analyze_sentence(sentence)
# print("Tokens:", analysis_result['tokens'])
# print("POS Tags:", analysis_result['pos_tags'])
# print("Named Entities:", analysis_result['named_entities'])
# print("Sentiment Scores:", analysis_result['sentiment_scores'])
# print("Positive Items:", analysis_result['positive_items'])
# print("Negative Items:", analysis_result['negative_items'])
# 
