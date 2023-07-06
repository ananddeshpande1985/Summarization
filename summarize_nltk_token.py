from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

checkpoint = "google/pegasus-xsum"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

long_text = "This is a very very long text. " * 300


sentences = nltk.tokenize.sent_tokenize(long_text)

# initialize
length = 0
chunk = ""
chunks = []
count = -1
for sentence in sentences:
  count += 1
  combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

  if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
    chunk += sentence + " " # add the sentence to the chunk
    length = combined_length # update the length counter

    # if it is the last sentence
    if count == len(sentences) - 1:
      chunks.append(chunk) # save the chunk
    
  else: 
    chunks.append(chunk) # save the chunk
    # reset 
    length = 0 
    chunk = ""

    # take care of the overflow sentence
    chunk += sentence + " "
    length = len(tokenizer.tokenize(sentence))

# inputs
inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]

# print summary
for input in inputs:
  output = model.generate(**input)
  print(tokenizer.decode(*output, skip_special_tokens=True))
